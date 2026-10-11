"""Configuration for the UrbanDreamer MAMBA baseline.

The architecture defaults intentionally match the upstream MAMBA paper code.
Smaller values can be supplied by unit tests and smoke runs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping


@dataclass(slots=True)
class MambaConfig:
    obs_dim: int
    action_dim: int = 4
    n_slots: int = 1
    device: str = "cuda"
    seed: int = 42

    hidden_size: int = 256
    model_hidden_size: int = 256
    embedding_size: int = 256
    deterministic_size: int = 256
    n_categoricals: int = 32
    n_classes: int = 32
    transformer_layers: int = 3
    transformer_heads: int = 8
    transformer_dropout: float = 0.1
    mlp_layers: int = 2

    replay_capacity_episodes: int = 20
    sequence_length: int = 20
    burn_in: int = 5
    max_agents_per_chunk: int = 32
    sequence_minibatch_size: int = 4
    min_replay_episodes: int = 1
    min_model_replay_frames: int = 100
    min_agent_replay_frames: int = 2500

    model_epochs: int = 1
    agent_epochs: int = 1
    critic_epochs: int = 5
    ppo_epochs: int = 5
    critic_warmup_updates: int = 20
    imagination_horizon: int = 15
    max_imagination_starts: int = 16
    imagination_alive_threshold: float = 0.5
    imagination_reward_scope: str = "individual"
    update_interval_episodes: int = 1

    model_lr: float = 2.0e-4
    actor_lr: float = 5.0e-4
    critic_lr: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip: float = 0.075
    target_policy_kl: float = 0.01
    entropy_coef: float = 1.0e-3
    value_coef: float = 0.5
    max_grad_norm: float = 100.0
    policy_max_grad_norm: float = 10.0
    critic_max_grad_norm: float = 10.0
    critic_huber_delta: float = 1.0
    target_tau: float = 0.05
    normalize_advantages: bool = True
    advantage_clip: float = 2.5

    kl_balance: float = 0.2
    free_nats: float = 0.0
    reconstruction_coef: float = 1.0
    reward_coef: float = 1.0
    continuation_coef: float = 1.0
    action_information_coef: float = 1.0
    availability_coef: float = 1.0
    availability_threshold: float = 0.5
    kl_coef: float = 1.0
    use_bfloat16: bool = True

    def __post_init__(self) -> None:
        integer_positive = (
            "obs_dim", "action_dim", "n_slots", "hidden_size",
            "model_hidden_size", "embedding_size", "deterministic_size",
            "n_categoricals", "n_classes", "transformer_layers",
            "transformer_heads", "replay_capacity_episodes",
            "sequence_length", "max_agents_per_chunk",
            "sequence_minibatch_size", "imagination_horizon",
            "max_imagination_starts", "model_epochs", "agent_epochs",
            "critic_epochs", "ppo_epochs", "update_interval_episodes",
        )
        for name in integer_positive:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.agent_epochs) != 1:
            raise ValueError(
                "agent_epochs must be 1; multi-rollout metric/state reducers are not interchangeable"
            )
        if not 0 <= self.burn_in < self.sequence_length:
            raise ValueError("burn_in must satisfy 0 <= burn_in < sequence_length")
        if self.hidden_size % self.transformer_heads:
            raise ValueError("hidden_size must be divisible by transformer_heads")
        if not 0.0 <= self.transformer_dropout < 1.0:
            raise ValueError("transformer_dropout must be in [0, 1)")
        for name in ("min_model_replay_frames", "min_agent_replay_frames"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.min_agent_replay_frames < self.min_model_replay_frames:
            raise ValueError(
                "min_agent_replay_frames must be at least min_model_replay_frames"
            )
        if int(self.critic_warmup_updates) < 0:
            raise ValueError("critic_warmup_updates must be non-negative")
        if not 0.0 <= self.target_tau <= 1.0:
            raise ValueError("target_tau must be in [0, 1]")
        if self.target_policy_kl < 0.0:
            raise ValueError("target_policy_kl must be non-negative")
        if not 0.0 < self.imagination_alive_threshold <= 1.0:
            raise ValueError("imagination_alive_threshold must be in (0, 1]")
        if not 0.0 < self.availability_threshold < 1.0:
            raise ValueError("availability_threshold must be in (0, 1)")
        if self.free_nats < 0.0:
            raise ValueError("free_nats must be non-negative")
        if not 0.0 < self.ppo_clip < 1.0:
            raise ValueError("ppo_clip must be in (0, 1)")
        if self.advantage_clip < 0.0:
            raise ValueError("advantage_clip must be non-negative")
        if self.critic_huber_delta <= 0.0:
            raise ValueError("critic_huber_delta must be positive")
        if self.policy_max_grad_norm <= 0.0 or self.critic_max_grad_norm <= 0.0:
            raise ValueError("policy and critic gradient norms must be positive")
        if self.imagination_reward_scope not in {"individual", "team"}:
            raise ValueError(
                "imagination_reward_scope must be 'individual' or 'team'"
            )
        for name in ("action_information_coef", "availability_coef"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not isinstance(self.normalize_advantages, bool):
            raise ValueError("normalize_advantages must be a boolean")

    @property
    def stochastic_size(self) -> int:
        return self.n_categoricals * self.n_classes

    @property
    def feature_size(self) -> int:
        return self.stochastic_size + self.deterministic_size

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "MambaConfig":
        names = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - names)
        if unknown:
            raise ValueError(f"unknown MambaConfig keys: {unknown}")
        return cls(**dict(values))

