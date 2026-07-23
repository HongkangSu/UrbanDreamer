"""Masked world-model and imagined-PPO learner for the SUMO MAMBA baseline."""

from __future__ import annotations

import os
import random
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import MambaConfig
from .networks import (
    MambaActor,
    MambaCritic,
    MambaWorldModel,
    RSSMState,
    initialize_module,
)
from .replay import (
    SparseBatch,
    SparseEpisode,
    SparseFrame,
    SparseReplayBuffer,
    fixed_slot_agent_chunks,
)


SCHEMA_VERSION = 3
MODEL_FAMILY = "mamba"


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(device=values.device, dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    numerator = (values * mask).sum()
    denominator = mask.expand_as(values).sum()
    return numerator / denominator.clamp_min(1.0)


def _masked_variance(values: Tensor, mask: Tensor) -> Tensor:
    mean = _masked_mean(values, mask)
    return _masked_mean((values - mean).square(), mask)


def _masked_explained_variance(target: Tensor, prediction: Tensor, mask: Tensor) -> Tensor:
    target_variance = _masked_variance(target, mask)
    residual_variance = _masked_variance(target - prediction, mask)
    return torch.where(
        target_variance > 1.0e-8,
        1.0 - residual_variance / target_variance.clamp_min(1.0e-8),
        torch.zeros_like(target_variance),
    )


def _optimizer_state_is_finite(optimizer: torch.optim.Optimizer) -> bool:
    """Return false before a poisoned Adam state can reach a later update."""
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all().item():
                return False
    return True


def _module_is_finite(module: nn.Module) -> bool:
    return all(
        torch.isfinite(parameter).all().item()
        for parameter in module.parameters()
    )


def _state_where(mask: Tensor, when_true: RSSMState, when_false: RSSMState) -> RSSMState:
    mask = mask.to(dtype=torch.bool)
    return RSSMState(
        logits=torch.where(mask.unsqueeze(-1).unsqueeze(-1), when_true.logits, when_false.logits),
        stoch=torch.where(mask.unsqueeze(-1), when_true.stoch, when_false.stoch),
        deter=torch.where(mask.unsqueeze(-1), when_true.deter, when_false.deter),
    )


def _zero_state_slots(state: RSSMState, reset: Tensor) -> RSSMState:
    keep = (~reset).to(state.deter.dtype)
    return RSSMState(
        logits=state.logits * keep.unsqueeze(-1).unsqueeze(-1),
        stoch=state.stoch * keep.unsqueeze(-1),
        deter=state.deter * keep.unsqueeze(-1),
    )


def observe_sparse_batch(
    model: MambaWorldModel,
    batch: SparseBatch,
    *,
    burn_in: int,
    sample: bool = True,
) -> tuple[RSSMState, RSSMState]:
    """Roll out posterior states while respecting presence and episode resets."""
    time_steps, batch_size, n_agents = batch.presence.shape
    state = model.initial_state(batch_size, n_agents, device=batch.observation.device)
    previous_action = torch.zeros(
        batch_size, n_agents, model.cfg.action_dim,
        device=batch.observation.device,
    )
    previous_presence = torch.zeros(
        batch_size, n_agents, dtype=torch.bool, device=batch.observation.device,
    )
    priors: list[RSSMState] = []
    posteriors: list[RSSMState] = []
    for time_index in range(time_steps):
        if time_index == burn_in and burn_in:
            state = state.detach()
            previous_action = previous_action.detach()
        presence = batch.presence[time_index]
        newly_present = presence & ~previous_presence
        if time_index:
            reset = newly_present | batch.terminal[time_index - 1]
            reset |= batch.truncated[time_index - 1].unsqueeze(-1)
            state = _zero_state_slots(state, reset)
        prior, posterior = model.observe(
            batch.observation[time_index],
            previous_action,
            state,
            presence,
            batch.obs_valid[time_index],
            sample=sample,
        )
        priors.append(prior)
        posteriors.append(posterior)
        state = posterior
        previous_action = batch.action[time_index] * batch.decision[time_index].unsqueeze(-1)
        previous_presence = presence
    return RSSMState.stack(priors), RSSMState.stack(posteriors)


def masked_world_model_loss(
    cfg: MambaConfig,
    model: MambaWorldModel,
    batch: SparseBatch,
) -> tuple[Tensor, dict[str, Tensor], RSSMState]:
    prior, posterior = observe_sparse_batch(
        model, batch, burn_in=cfg.burn_in, sample=True,
    )
    features = posterior.features
    reconstruction = model.observation_decoder(features)

    # Rewards and continuation are consequences of the recorded action.  Roll
    # one prior step from every posterior (batched over time) instead of asking
    # a state-only head to explain rewards produced by different route choices.
    time_steps, batch_size, n_agents = batch.presence.shape
    flattened_state = RSSMState(
        logits=posterior.logits.flatten(0, 1),
        stoch=posterior.stoch.flatten(0, 1),
        deter=posterior.deter.flatten(0, 1),
    )
    predicted_next = model.transition(
        batch.action.flatten(0, 1),
        flattened_state,
        batch.presence.flatten(0, 1),
        sample=True,
    )
    # Upstream MAMBA predicts action consequences from the current stochastic
    # state and the action-conditioned next deterministic state.  Using the
    # newly sampled prior stochastic state here adds target noise and changes
    # the reward/continuation semantics used by imagined rollouts.
    predicted_next_deter = predicted_next.deter.reshape(
        time_steps, batch_size, n_agents, -1,
    )
    consequence_features = torch.cat(
        (posterior.stoch, predicted_next_deter), dim=-1,
    )
    reward_prediction = model.reward_model(consequence_features)
    continuation_logits = model.continuation_model(consequence_features)
    action_information_logits = model.action_information_model(consequence_features)
    availability_logits = model.availability_model(posterior.features)

    observation_mask = batch.loss_mask & batch.obs_valid
    decision_mask = batch.loss_mask & batch.decision
    temporal = (
        torch.arange(batch.presence.shape[0], device=batch.presence.device).view(-1, 1, 1)
        >= cfg.burn_in
    )
    # A terminal reported in frame t+1 means the vehicle disappeared after its
    # state/action in frame t.  Do not invent a continuation label at the end of
    # a sampled window, where no next boundary is available.
    next_terminal = torch.zeros_like(batch.terminal)
    next_terminal[:-1] = batch.terminal[1:]
    next_truncated = torch.zeros_like(batch.truncated)
    next_truncated[:-1] = batch.truncated[1:]
    has_next_step = torch.zeros_like(batch.valid_step)
    has_next_step[:-1] = batch.valid_step[1:]
    base_continuation_mask = (
        temporal & has_next_step.unsqueeze(-1) & batch.presence
    )

    reconstruction_error = F.smooth_l1_loss(
        reconstruction, batch.observation, reduction="none",
    ).sum(dim=-1)
    reward_error = F.smooth_l1_loss(
        reward_prediction, batch.reward, reduction="none",
    ).squeeze(-1)
    # A time-limit truncation is censored experience, not evidence that an
    # unfinished vehicle arrived.  Labelling it terminal teaches the actor to
    # seek episode cutoffs because a constant step cost then appears to stop.
    # Preserve genuine disappearances reported at the same final boundary and
    # exclude only unresolved slots from the continuation likelihood.
    terminal_target = next_terminal
    unresolved_truncation = next_truncated.unsqueeze(-1) & ~next_terminal
    continuation_mask = base_continuation_mask & ~unresolved_truncation
    continuation_target = (~terminal_target).to(continuation_logits.dtype)
    continuation_error = F.binary_cross_entropy_with_logits(
        continuation_logits.squeeze(-1), continuation_target, reduction="none",
    )
    action_index = batch.action.argmax(dim=-1)
    action_information_error = F.cross_entropy(
        action_information_logits.reshape(-1, cfg.action_dim),
        action_index.reshape(-1),
        reduction="none",
    ).reshape_as(action_index)
    availability_error = F.binary_cross_entropy_with_logits(
        availability_logits,
        batch.action_valid.to(availability_logits.dtype),
        reduction="none",
    ).mean(dim=-1)

    posterior_probability = posterior.logits.softmax(dim=-1)
    prior_probability = prior.logits.softmax(dim=-1)
    eps = torch.finfo(posterior_probability.dtype).eps

    def categorical_kl(left: Tensor, right: Tensor) -> Tensor:
        return (
            left * ((left + eps).log() - (right + eps).log())
        ).sum(dim=-1).mean(dim=-1)

    posterior_kl = categorical_kl(posterior_probability, prior_probability.detach())
    prior_kl = categorical_kl(posterior_probability.detach(), prior_probability)
    divergence = cfg.kl_balance * posterior_kl + (1.0 - cfg.kl_balance) * prior_kl
    if cfg.free_nats > 0:
        divergence = divergence.clamp_min(cfg.free_nats)

    reconstruction_loss = _masked_mean(reconstruction_error, observation_mask)
    reward_loss = _masked_mean(reward_error, decision_mask)
    continue_mask = continuation_mask & ~terminal_target
    terminal_mask = continuation_mask & terminal_target
    truncated_excluded_mask = base_continuation_mask & unresolved_truncation
    continue_count = continue_mask.sum()
    terminal_count = terminal_mask.sum()
    continue_loss = _masked_mean(continuation_error, continue_mask)
    terminal_loss = _masked_mean(continuation_error, terminal_mask)
    # Keep the Bernoulli probability calibrated because imagination consumes it
    # directly as a continuation discount.  Tail-window sampling restores the
    # sparse terminal events to their empirical frequency; forcing a 50/50
    # class loss here would instead make an uninformative model predict 0.5.
    continuation_loss = _masked_mean(continuation_error, continuation_mask)
    action_information_loss = _masked_mean(action_information_error, decision_mask)
    availability_loss = _masked_mean(availability_error, decision_mask)
    kl_loss = _masked_mean(divergence, observation_mask)
    loss = (
        cfg.reconstruction_coef * reconstruction_loss
        + cfg.reward_coef * reward_loss
        + cfg.continuation_coef * continuation_loss
        + cfg.action_information_coef * action_information_loss
        + cfg.availability_coef * availability_loss
        + cfg.kl_coef * kl_loss
    )
    continuation_probability = continuation_logits.sigmoid().squeeze(-1)
    continuation_count = continue_count + terminal_count
    metrics = {
        "model_loss": loss.detach(),
        "reconstruction_loss": reconstruction_loss.detach(),
        "reward_loss": reward_loss.detach(),
        "continuation_loss": continuation_loss.detach(),
        "continuation_continue_loss": continue_loss.detach(),
        "continuation_terminal_loss": terminal_loss.detach(),
        "continuation_terminal_fraction": (
            terminal_count.to(torch.float32)
            / continuation_count.to(torch.float32).clamp_min(1.0)
        ).detach(),
        "continuation_continue_probability": _masked_mean(
            continuation_probability, continue_mask,
        ).detach(),
        "continuation_terminal_probability": _masked_mean(
            continuation_probability, terminal_mask,
        ).detach(),
        "continuation_continue_count": continue_count.detach().to(torch.float32),
        "continuation_terminal_count": terminal_count.detach().to(torch.float32),
        "continuation_truncated_count": truncated_excluded_mask.sum().detach().to(
            torch.float32
        ),
        "continuation_truncated_excluded_count": truncated_excluded_mask.sum().detach().to(
            torch.float32
        ),
        "action_information_loss": action_information_loss.detach(),
        "availability_loss": availability_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "model_observations": observation_mask.sum().detach().to(torch.float32),
        "model_decisions": decision_mask.sum().detach().to(torch.float32),
    }
    return loss, metrics, posterior


def _sample_decision_starts(
    state: RSSMState,
    batch: SparseBatch,
    *,
    max_starts: int,
) -> tuple[RSSMState, Tensor, Tensor]:
    """Sample joint imagination starts from every real decision time."""
    _, batch_size, _ = batch.decision.shape
    supervised_decision = batch.decision & batch.loss_mask
    candidates = torch.nonzero(supervised_decision.any(dim=-1), as_tuple=False)
    if len(candidates) > max_starts:
        permutation = torch.randperm(
            len(candidates), device=candidates.device,
        )[:max_starts]
        candidates = candidates.index_select(0, permutation)

    def gather(value: Tensor, flat_indices: Tensor) -> Tensor:
        return value.flatten(0, 1).index_select(0, flat_indices)

    if len(candidates):
        flat_indices = candidates[:, 0] * batch_size + candidates[:, 1]
    else:
        flat_indices = torch.empty(
            (0,), dtype=torch.long, device=batch.decision.device,
        )

    selected = RSSMState(
        logits=gather(state.logits, flat_indices),
        stoch=gather(state.stoch, flat_indices),
        deter=gather(state.deter, flat_indices),
    )
    active = gather(supervised_decision, flat_indices)
    action_valid = gather(batch.action_valid, flat_indices)
    action_valid = torch.where(
        active.unsqueeze(-1), action_valid, torch.ones_like(action_valid),
    )
    return selected, active, action_valid


class MambaLearner:
    """MAMBA learner plus a fixed-global-slot online controller."""

    def __init__(
        self,
        config: MambaConfig,
        machine_agent_ids: Sequence[str] | None = None,
    ) -> None:
        self.config = config
        if machine_agent_ids is None:
            machine_agent_ids = tuple(str(index) for index in range(config.n_slots))
        self.machine_agent_ids = tuple(str(value) for value in machine_agent_ids)
        if len(self.machine_agent_ids) != config.n_slots:
            raise ValueError("machine_agent_ids length must equal config.n_slots")
        if len(set(self.machine_agent_ids)) != len(self.machine_agent_ids):
            raise ValueError("machine_agent_ids must be unique")

        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)

        self.world_model = MambaWorldModel(config).to(self.device)
        self.actor = MambaActor(config).to(self.device)
        self.critic = MambaCritic(config).to(self.device)
        initialize_module(self.world_model)
        initialize_module(self.actor, orthogonal=True)
        initialize_module(self.critic)
        # A random value head contaminates the first imagined lambda returns via
        # both the bootstrap and the baseline.  Start from V=0 so critic-only
        # warm-up is a finite-horizon reward fit instead of random self-bootstrapping.
        critic_output = self.critic.value[-1]
        if not isinstance(critic_output, nn.Linear):
            raise TypeError("MambaCritic.value must end in a linear output layer")
        nn.init.zeros_(critic_output.weight)
        nn.init.zeros_(critic_output.bias)
        self.target_critic = deepcopy(self.critic).to(self.device).eval()
        for parameter in self.target_critic.parameters():
            parameter.requires_grad_(False)

        self.model_optimizer = torch.optim.Adam(
            self.world_model.parameters(), lr=config.model_lr,
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_lr, weight_decay=1.0e-5,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_lr,
        )
        self.replay = SparseReplayBuffer(
            capacity_episodes=config.replay_capacity_episodes,
            obs_dim=config.obs_dim,
            action_dim=config.action_dim,
            seed=config.seed,
        )
        self.update_count = 0
        self.critic_update_count = 0
        self._critic_warmup_synced = config.critic_warmup_updates == 0
        self.episodes_seen = 0
        self.environment_steps = 0
        # Unlike ``replay.n_frames``, this counter is monotonic when the
        # episode-bounded replay evicts old experience.  Warm-up thresholds
        # describe how much experience has been collected, not how many
        # frames happen to remain resident in the buffer.
        self.collected_frames = 0
        self._current_episode: SparseEpisode | None = None
        self._decision_open = False
        self._current_presence = torch.zeros(config.n_slots, dtype=torch.bool, device=self.device)
        self._previous_presence = torch.zeros_like(self._current_presence)
        self._controller_state = self.world_model.initial_state(
            1, config.n_slots, device=self.device,
        )
        self._controller_prior = self._controller_state
        self._controller_previous_action = torch.zeros(
            1, config.n_slots, config.action_dim, device=self.device,
        )
        self.world_model.eval()
        self.actor.eval()
        self.critic.eval()

    def _autocast(self):
        if self.device.type == "cuda" and self.config.use_bfloat16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def reset_episode(self) -> None:
        if self._current_episode is not None and self._current_episode.frames:
            raise RuntimeError("end_episode must be called before resetting a non-empty episode")
        self._current_episode = SparseEpisode(self.episodes_seen + 1)
        self._decision_open = False
        self._current_presence.zero_()
        self._previous_presence.zero_()
        self._controller_state = self.world_model.initial_state(
            1, self.config.n_slots, device=self.device,
        )
        self._controller_prior = self._controller_state
        self._controller_previous_action.zero_()

    @torch.no_grad()
    def begin_decision(self, presence_slots: Sequence[int]) -> None:
        if self._decision_open:
            raise RuntimeError("end_decision must be called before begin_decision")
        slots = [int(slot) for slot in presence_slots]
        if len(slots) != len(set(slots)):
            raise ValueError("presence_slots contains duplicates")
        if any(slot < 0 or slot >= self.config.n_slots for slot in slots):
            raise IndexError("presence slot is outside configured global slots")
        if self._current_episode is None:
            self.reset_episode()

        self.world_model.eval()
        presence = torch.zeros_like(self._current_presence)
        if slots:
            presence[torch.as_tensor(slots, device=self.device)] = True
        newly_present = presence & ~self._previous_presence
        reset = newly_present | ~presence
        previous_state = _zero_state_slots(self._controller_state, reset.unsqueeze(0))
        previous_action = self._controller_previous_action.clone()
        previous_action[:, reset] = 0
        next_state = self.world_model.initial_state(
            1, self.config.n_slots, device=self.device,
        )
        for slot_block in fixed_slot_agent_chunks(
            slots, self.config.max_agents_per_chunk,
        ):
            if not len(slot_block):
                continue
            chunk = torch.as_tensor(
                slot_block,
                dtype=torch.long,
                device=self.device,
            )
            chunk_state = RSSMState(
                logits=previous_state.logits.index_select(1, chunk),
                stoch=previous_state.stoch.index_select(1, chunk),
                deter=previous_state.deter.index_select(1, chunk),
            )
            chunk_prior = self.world_model.transition(
                previous_action.index_select(1, chunk),
                chunk_state,
                torch.ones((1, len(chunk)), dtype=torch.bool, device=self.device),
                sample=True,
            )
            next_state.logits[:, chunk] = chunk_prior.logits
            next_state.stoch[:, chunk] = chunk_prior.stoch
            next_state.deter[:, chunk] = chunk_prior.deter
        self._controller_prior = next_state
        self._controller_state = RSSMState(
            logits=next_state.logits.clone(),
            stoch=next_state.stoch.clone(),
            deter=next_state.deter.clone(),
        )
        self._current_presence = presence
        self._decision_open = True

    @torch.no_grad()
    def act(
        self,
        slot: int,
        obs: np.ndarray | Tensor,
        action_mask: np.ndarray | Tensor,
        deterministic: bool = False,
    ) -> dict[str, Any]:
        if not self._decision_open:
            raise RuntimeError("begin_decision must be called before act")
        slot = int(slot)
        if slot < 0 or slot >= self.config.n_slots or not self._current_presence[slot]:
            raise ValueError("act slot must be present in the open decision")
        observation = torch.as_tensor(obs, dtype=torch.float32, device=self.device).reshape(1, 1, -1)
        if observation.shape[-1] != self.config.obs_dim:
            raise ValueError("observation dimension does not match MambaConfig.obs_dim")
        valid = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).reshape(1, 1, -1)
        if valid.shape[-1] != self.config.action_dim:
            raise ValueError("action mask dimension does not match MambaConfig.action_dim")
        if not valid.any():
            raise ValueError("act requires at least one valid action")
        index = torch.as_tensor([slot], dtype=torch.long, device=self.device)
        prior = RSSMState(
            logits=self._controller_prior.logits.index_select(1, index),
            stoch=self._controller_prior.stoch.index_select(1, index),
            deter=self._controller_prior.deter.index_select(1, index),
        )
        posterior = self.world_model.posterior_from_prior(
            prior,
            observation,
            torch.ones((1, 1), dtype=torch.bool, device=self.device),
            sample=not deterministic,
        )
        self._controller_state.logits[:, slot:slot + 1] = posterior.logits
        self._controller_state.stoch[:, slot:slot + 1] = posterior.stoch
        self._controller_state.deter[:, slot:slot + 1] = posterior.deter
        distribution = self.actor.distribution(posterior.features, valid)
        action = distribution.logits.argmax(dim=-1) if deterministic else distribution.sample()
        return {
            "action": int(action.item()),
            "log_prob": float(distribution.log_prob(action).item()),
            "entropy": float(distribution.entropy().item()),
            "logits": distribution.logits.squeeze(0).squeeze(0).float().cpu().numpy(),
        }

    def end_decision(self, frame: Mapping[str, Any]) -> None:
        if not self._decision_open:
            raise RuntimeError("begin_decision must be called before end_decision")
        sparse = SparseFrame.from_mapping(
            frame, obs_dim=self.config.obs_dim, action_dim=self.config.action_dim,
        )
        expected = set(torch.nonzero(self._current_presence, as_tuple=False).flatten().cpu().tolist())
        if set(sparse.presence_slots.tolist()) != expected:
            raise ValueError("frame presence_slots differs from begin_decision")
        assert self._current_episode is not None
        if not self._current_episode.frames:
            self._current_episode.episode = sparse.episode
        self._current_episode.append(sparse)
        self._controller_previous_action.zero_()
        if len(sparse.decision_slots):
            slots = torch.as_tensor(sparse.decision_slots, dtype=torch.long, device=self.device)
            actions = torch.as_tensor(sparse.actions, dtype=torch.long, device=self.device)
            self._controller_previous_action[0, slots] = F.one_hot(
                actions, self.config.action_dim,
            ).to(self._controller_previous_action.dtype)
        if len(sparse.terminal_slots):
            terminal = torch.as_tensor(sparse.terminal_slots, dtype=torch.long, device=self.device)
            mask = torch.zeros((1, self.config.n_slots), dtype=torch.bool, device=self.device)
            mask[:, terminal] = True
            self._controller_state = _zero_state_slots(self._controller_state, mask)
            self._controller_previous_action[:, terminal] = 0
        self._previous_presence = self._current_presence.clone()
        self._decision_open = False
        self.environment_steps += 1
        self.collected_frames += 1

    def end_episode(self, frame: Mapping[str, Any] | None = None) -> None:
        if frame is not None:
            if self._decision_open:
                self.end_decision(frame)
            else:
                sparse = SparseFrame.from_mapping(
                    frame, obs_dim=self.config.obs_dim, action_dim=self.config.action_dim,
                )
                if self._current_episode is None:
                    self._current_episode = SparseEpisode(sparse.episode)
                self._current_episode.append(sparse)
                self.collected_frames += 1
        if self._decision_open:
            raise RuntimeError("an open decision requires a final frame")
        if self._current_episode is not None and self._current_episode.frames:
            self.replay.add_episode(self._current_episode)
            self.episodes_seen += 1
        self._current_episode = None
        self._current_presence.zero_()
        self._previous_presence.zero_()
        self._controller_previous_action.zero_()

    def _sample_batch(self) -> SparseBatch:
        return self.replay.sample(
            batch_size=self.config.sequence_minibatch_size,
            sequence_length=self.config.sequence_length,
            burn_in=self.config.burn_in,
            max_agents_per_chunk=self.config.max_agents_per_chunk,
            device=self.device,
        )

    def _train_world_model(self, batch: SparseBatch) -> dict[str, float]:
        self.world_model.train()
        self.model_optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            loss, tensors, _ = masked_world_model_loss(self.config, self.world_model, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite MAMBA world-model loss")
        loss.backward()
        try:
            gradient = nn.utils.clip_grad_norm_(
                self.world_model.parameters(),
                self.config.max_grad_norm,
                error_if_nonfinite=True,
            )
        except RuntimeError as exc:
            raise FloatingPointError(
                "non-finite MAMBA world-model gradient"
            ) from exc
        if not torch.isfinite(torch.as_tensor(gradient)):
            raise FloatingPointError("non-finite MAMBA world-model gradient")
        self.model_optimizer.step()
        if not _module_is_finite(self.world_model) or not _optimizer_state_is_finite(
            self.model_optimizer
        ):
            # Abort this process before a poisoned model can generate imagined
            # targets.  The last atomic checkpoint remains safe for resuming.
            raise FloatingPointError("non-finite MAMBA world-model optimizer step")
        self.world_model.eval()
        result = {name: float(value.float().item()) for name, value in tensors.items()}
        result["model_grad_norm"] = float(torch.as_tensor(gradient).float().item())
        return result

    @torch.no_grad()
    def _imagination_rollout(self, batch: SparseBatch) -> dict[str, Tensor] | None:
        _, posterior = observe_sparse_batch(
            self.world_model, batch, burn_in=self.config.burn_in, sample=True,
        )
        state, active, action_valid = _sample_decision_starts(
            posterior,
            batch,
            max_starts=self.config.max_imagination_starts,
        )
        if not active.any():
            return None
        states: list[Tensor] = []
        actions: list[Tensor] = []
        old_log_probs: list[Tensor] = []
        old_logits: list[Tensor] = []
        entropies: list[Tensor] = []
        rewards: list[Tensor] = []
        continuations: list[Tensor] = []
        target_values: list[Tensor] = []
        presence_masks: list[Tensor] = []
        survival_weights: list[Tensor] = []
        action_valid_masks: list[Tensor] = []
        initial_active = active.clone()
        reach_probability = initial_active.to(dtype=state.deter.dtype)
        loss_weight = reach_probability.clone()
        presence = initial_active.clone()
        initial_agent_count = initial_active.sum(dim=-1, keepdim=True).to(loss_weight.dtype)
        initial_agent_count = initial_agent_count.clamp_min(1.0)
        for _ in range(self.config.imagination_horizon):
            # A predicted terminal should remove the slot from subsequent
            # imagined communication and value losses.  Keep the soft
            # continuation probability as a weight for expected returns, but
            # use a deterministic presence threshold for the RSSM attention
            # mask so an arrived vehicle cannot keep influencing survivors.
            action_valid_step = torch.where(
                presence.unsqueeze(-1), action_valid, torch.ones_like(action_valid),
            )
            features = state.features
            distribution = self.actor.distribution(features, action_valid_step)
            action_index = distribution.sample()
            one_hot = F.one_hot(action_index, self.config.action_dim).to(features.dtype)
            states.append(features)
            actions.append(action_index)
            old_log_probs.append(distribution.log_prob(action_index))
            old_logits.append(distribution.logits)
            entropies.append(distribution.entropy())
            target_values.append(self.target_critic(features, presence).squeeze(-1))
            presence_masks.append(presence)
            survival_weights.append(loss_weight)
            action_valid_masks.append(action_valid_step)
            current_stoch = state.stoch
            state = self.world_model.transition(one_hot, state, presence, sample=True)
            consequence_features = torch.cat((current_stoch, state.deter), dim=-1)
            individual_reward = self.world_model.reward_model(
                consequence_features,
            ).squeeze(-1)
            masked_individual_reward = (
                individual_reward * presence.to(individual_reward.dtype)
            )
            if self.config.imagination_reward_scope == "team":
                # Upstream MAMBA broadcasts a cooperative team mean.  Retain it
                # as an explicit compatibility mode, but SUMO defaults to local
                # vehicle rewards so one route decision is not diluted by 1/N.
                team_reward = (
                    masked_individual_reward.sum(dim=-1, keepdim=True)
                    / initial_agent_count.to(individual_reward.dtype)
                )
                reward_step = team_reward.expand_as(individual_reward)
            else:
                reward_step = masked_individual_reward
            rewards.append(reward_step)
            continuation = self.world_model.continuation_model(
                consequence_features,
            ).sigmoid().squeeze(-1) * presence.to(individual_reward.dtype)
            continuations.append(continuation)
            reach_probability = reach_probability * continuation
            loss_weight = loss_weight * self.config.gamma * continuation
            presence = initial_active & (
                reach_probability >= float(self.config.imagination_alive_threshold)
            )
            reach_probability = reach_probability * presence.to(reach_probability.dtype)
            loss_weight = loss_weight * presence.to(loss_weight.dtype)
            availability_logits = self.world_model.availability_model(state.features)
            predicted_action_valid = (
                availability_logits.sigmoid() >= float(self.config.availability_threshold)
            )
            # The current-to-destination generator always exposes candidate 0
            # for a recorded decision.  Keeping it available prevents an
            # imperfect imagined availability head from deleting the safe
            # shortest-freeflow anchor and forcing a worse alternative.
            predicted_action_valid[..., 0] = True
            no_predicted_action = ~predicted_action_valid.any(dim=-1, keepdim=True)
            fallback_action = availability_logits.argmax(dim=-1)
            fallback_mask = F.one_hot(
                fallback_action, self.config.action_dim,
            ).to(torch.bool)
            predicted_action_valid = torch.where(
                no_predicted_action,
                fallback_mask,
                predicted_action_valid,
            )
            action_valid = torch.where(
                presence.unsqueeze(-1),
                predicted_action_valid,
                torch.ones_like(predicted_action_valid),
            )
        bootstrap = self.target_critic(state.features, presence).squeeze(-1)
        reward = torch.stack(rewards)
        continuation = torch.stack(continuations)
        value = torch.stack(target_values)
        returns = torch.empty_like(reward)
        accumulated = bootstrap
        for time_index in reversed(range(self.config.imagination_horizon)):
            next_value = bootstrap if time_index == self.config.imagination_horizon - 1 else value[time_index + 1]
            one_step = reward[time_index] + self.config.gamma * continuation[time_index] * (
                (1.0 - self.config.gae_lambda) * next_value
                + self.config.gae_lambda * accumulated
            )
            returns[time_index] = one_step
            accumulated = one_step
        return {
            "features": torch.stack(states),
            "actions": torch.stack(actions),
            "old_log_probs": torch.stack(old_log_probs),
            "old_logits": torch.stack(old_logits),
            "old_entropies": torch.stack(entropies),
            "returns": returns,
            # Keep ``active`` as the immutable start mask for callers that
            # report per-start samples; ``presence`` is the dynamic mask used
            # by the critic and RSSM communication after predicted arrivals.
            "active": initial_active.unsqueeze(0).expand(
                self.config.imagination_horizon, -1, -1,
            ),
            "presence": torch.stack(presence_masks),
            "initial_active": initial_active.unsqueeze(0).expand(
                self.config.imagination_horizon, -1, -1,
            ),
            "survival": torch.stack(survival_weights),
            "action_valid": torch.stack(action_valid_masks),
            "imagined_reward": reward,
            "imagined_starts": torch.as_tensor(
                state.logits.shape[0], dtype=torch.float32, device=reward.device,
            ),
        }

    def _train_agent(
        self,
        batch: SparseBatch,
        *,
        update_actor: bool = True,
    ) -> dict[str, float]:
        """Fit the critic on imagined returns and optionally perform PPO.

        Critic-only calls are used during world-model/critic warm-up.  Once the
        actor is enabled, its advantage is computed exactly once from the
        *pre-update* critic, as required by PPO; critic epochs remain independent
        from PPO early stopping and cannot shorten when a policy step is rejected.
        """
        self.world_model.eval()
        rollout = self._imagination_rollout(batch)
        if rollout is None:
            return {
                "agent_skipped": 1.0,
                "actor_skipped": 1.0,
                "critic_skipped": 1.0,
                "imagined_samples": 0.0,
            }
        features = rollout["features"].detach()
        actions = rollout["actions"].detach()
        old_log_probs = rollout["old_log_probs"].detach()
        old_logits = rollout["old_logits"].detach()
        returns = rollout["returns"].detach()
        presence = rollout["presence"].detach()
        initial_active = rollout["initial_active"].detach()
        survival = rollout["survival"].detach()
        action_valid = rollout["action_valid"].detach()
        horizon, batch_size, n_agents, feature_size = features.shape
        flat_features = features.reshape(horizon * batch_size, n_agents, feature_size)
        flat_presence = presence.reshape(horizon * batch_size, n_agents)

        # Freeze a pre-update baseline for all PPO epochs.  During production the
        # critic has already received critic-only warm-up updates; it is never a
        # random baseline when update_actor=True.
        with torch.no_grad():
            baseline_values = self.critic(flat_features, flat_presence).reshape(
                horizon, batch_size, n_agents,
            )
        raw_advantages = returns - baseline_values
        raw_advantage_mean = _masked_mean(raw_advantages, survival)
        raw_advantage_variance = _masked_variance(raw_advantages, survival)
        advantages = raw_advantages
        if self.config.normalize_advantages and survival.sum().item() > 1.0:
            advantages = (advantages - raw_advantage_mean) / (
                raw_advantage_variance.sqrt() + 1.0e-5
            )
        if self.config.advantage_clip > 0.0:
            advantages = advantages.clamp(
                min=-float(self.config.advantage_clip),
                max=float(self.config.advantage_clip),
            )
        weighted_advantage_mean = _masked_mean(advantages, survival)
        weighted_advantage_std = _masked_variance(advantages, survival).sqrt()
        live_advantages = advantages[survival > 0]
        advantage_min = float(live_advantages.min().item()) if live_advantages.numel() else 0.0
        advantage_max = float(live_advantages.max().item()) if live_advantages.numel() else 0.0
        critic_explained_variance_pre = _masked_explained_variance(
            returns, baseline_values, survival,
        )

        actor_loss_value = entropy_value = 0.0
        policy_kl_value = clip_fraction_value = 0.0
        final_policy_kl = max_policy_kl = max_attempted_policy_kl = 0.0
        rejected_policy_kl = 0.0
        actor_grad_norm_value = 0.0
        ppo_epochs_completed = ppo_epochs_attempted = ppo_epochs_rejected = 0
        ppo_nonfinite_rejected = 0

        if update_actor:
            for _ in range(self.config.ppo_epochs):
                # A post-step KL check without rollback leaves the overshooting
                # update installed.  Snapshot both parameters and Adam moments so
                # a rejected epoch is genuinely side-effect free.
                actor_before = deepcopy(self.actor.state_dict())
                optimizer_before = deepcopy(self.actor_optimizer.state_dict())
                self.actor.train()
                self.actor_optimizer.zero_grad(set_to_none=True)
                distribution = self.actor.distribution(features, action_valid)
                new_log_prob = distribution.log_prob(actions)
                ratio = (new_log_prob - old_log_probs).exp()
                unclipped = ratio * advantages
                clipped = ratio.clamp(
                    1.0 - self.config.ppo_clip, 1.0 + self.config.ppo_clip,
                ) * advantages
                objective = torch.minimum(unclipped, clipped)
                entropy = distribution.entropy()
                actor_loss = -_masked_mean(
                    objective, survival,
                ) - self.config.entropy_coef * _masked_mean(entropy, survival)
                if not torch.isfinite(actor_loss):
                    raise FloatingPointError("non-finite MAMBA actor loss")
                actor_loss.backward()
                actor_gradient = nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.config.policy_max_grad_norm,
                )
                if not torch.isfinite(torch.as_tensor(actor_gradient)):
                    ppo_epochs_attempted += 1
                    ppo_epochs_rejected += 1
                    ppo_nonfinite_rejected += 1
                    self.actor.load_state_dict(actor_before)
                    self.actor_optimizer.load_state_dict(optimizer_before)
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    break
                self.actor_optimizer.step()
                ppo_epochs_attempted += 1

                policy_metrics_finite = all(
                    torch.isfinite(parameter).all().item()
                    for parameter in self.actor.parameters()
                )
                try:
                    with torch.no_grad():
                        updated_distribution = self.actor.distribution(features, action_valid)
                        updated_log_prob = updated_distribution.log_prob(actions)
                        updated_ratio = (updated_log_prob - old_log_probs).exp()
                        old_probability = old_logits.exp()
                        categorical_kl = (
                            old_probability
                            * (old_logits - updated_distribution.logits)
                        ).sum(dim=-1).clamp_min(0.0)
                        current_policy_kl = _masked_mean(categorical_kl, survival)
                        current_clip_fraction = _masked_mean(
                            (
                                (updated_ratio - 1.0).abs() > self.config.ppo_clip
                            ).to(survival.dtype),
                            survival,
                        )
                    policy_metrics_finite = policy_metrics_finite and bool(
                        torch.isfinite(current_policy_kl).item()
                        and torch.isfinite(current_clip_fraction).item()
                        and torch.isfinite(updated_log_prob).all().item()
                    )
                except (RuntimeError, ValueError):
                    policy_metrics_finite = False
                if not policy_metrics_finite:
                    ppo_epochs_rejected += 1
                    ppo_nonfinite_rejected += 1
                    self.actor.load_state_dict(actor_before)
                    self.actor_optimizer.load_state_dict(optimizer_before)
                    break
                attempted_kl = float(current_policy_kl.item())
                max_attempted_policy_kl = max(max_attempted_policy_kl, attempted_kl)
                if (
                    self.config.target_policy_kl > 0.0
                    and attempted_kl > self.config.target_policy_kl
                ):
                    rejected_policy_kl = attempted_kl
                    ppo_epochs_rejected += 1
                    self.actor.load_state_dict(actor_before)
                    self.actor_optimizer.load_state_dict(optimizer_before)
                    break

                actor_loss_value += float(actor_loss.detach().item())
                actor_grad_norm_value += float(
                    torch.as_tensor(actor_gradient).float().item()
                )
                entropy_value += float(_masked_mean(
                    updated_distribution.entropy(), survival,
                ).item())
                policy_kl_value += attempted_kl
                final_policy_kl = attempted_kl
                max_policy_kl = max(max_policy_kl, attempted_kl)
                clip_fraction_value += float(current_clip_fraction.item())
                ppo_epochs_completed += 1

        # Critic epochs are deliberately separate from PPO epochs.  They run in
        # full even when the actor is warming up or a policy epoch is rejected.
        critic_loss_value = critic_grad_norm_value = 0.0
        for _ in range(self.config.critic_epochs):
            self.critic.train()
            self.critic_optimizer.zero_grad(set_to_none=True)
            predicted = self.critic(flat_features, flat_presence).reshape(
                horizon, batch_size, n_agents,
            )
            critic_error = F.smooth_l1_loss(
                predicted,
                returns,
                reduction="none",
                beta=float(self.config.critic_huber_delta),
            )
            critic_loss = self.config.value_coef * _masked_mean(
                critic_error, survival,
            )
            if not torch.isfinite(critic_loss):
                raise FloatingPointError("non-finite MAMBA critic loss")
            critic_loss.backward()
            try:
                critic_gradient = nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    self.config.critic_max_grad_norm,
                    error_if_nonfinite=True,
                )
            except RuntimeError as exc:
                raise FloatingPointError("non-finite MAMBA critic gradient") from exc
            if not torch.isfinite(torch.as_tensor(critic_gradient)):
                raise FloatingPointError("non-finite MAMBA critic gradient")
            self.critic_optimizer.step()
            if not _module_is_finite(self.critic) or not _optimizer_state_is_finite(
                self.critic_optimizer
            ):
                # In particular, do not let this critic reach the hard/Polyak
                # target synchronization below.
                raise FloatingPointError("non-finite MAMBA critic optimizer step")
            critic_loss_value += float(critic_loss.detach().item())
            critic_grad_norm_value += float(
                torch.as_tensor(critic_gradient).float().item()
            )

        self.critic_update_count += 1
        target_hard_sync = 0.0
        with torch.no_grad():
            if (
                not self._critic_warmup_synced
                and self.critic_update_count >= self.config.critic_warmup_updates
            ):
                self.target_critic.load_state_dict(self.critic.state_dict())
                self._critic_warmup_synced = True
                target_hard_sync = 1.0
            elif self._critic_warmup_synced:
                for target, source in zip(
                    self.target_critic.parameters(), self.critic.parameters(), strict=True,
                ):
                    if self.config.target_tau >= 1.0:
                        target.copy_(source)
                    else:
                        target.lerp_(source, self.config.target_tau)

        self.actor.eval()
        self.critic.eval()
        self.target_critic.eval()
        with torch.no_grad():
            fitted_values = self.critic(flat_features, flat_presence).reshape(
                horizon, batch_size, n_agents,
            )
            critic_explained_variance_post = _masked_explained_variance(
                returns, fitted_values, survival,
            )

        actor_updated = update_actor and ppo_epochs_completed > 0
        actor_denominator = float(max(1, ppo_epochs_completed))
        critic_denominator = float(max(1, self.config.critic_epochs))
        return {
            "agent_skipped": float(not actor_updated),
            "actor_skipped": float(not actor_updated),
            "actor_updated": float(actor_updated),
            "critic_skipped": 0.0,
            "actor_loss": actor_loss_value / actor_denominator,
            "critic_loss": critic_loss_value / critic_denominator,
            "actor_grad_norm": actor_grad_norm_value / actor_denominator,
            "critic_grad_norm": critic_grad_norm_value / critic_denominator,
            "policy_entropy": entropy_value / actor_denominator,
            "policy_kl": policy_kl_value / actor_denominator,
            "policy_kl_final": final_policy_kl,
            "policy_kl_max": max_policy_kl,
            "policy_kl_max_attempted": max_attempted_policy_kl,
            "policy_kl_rejected": rejected_policy_kl,
            "clip_fraction": clip_fraction_value / actor_denominator,
            "ppo_epochs_completed": float(ppo_epochs_completed),
            "ppo_epochs_attempted": float(ppo_epochs_attempted),
            "ppo_epochs_rejected": float(ppo_epochs_rejected),
            "ppo_nonfinite_rejected": float(ppo_nonfinite_rejected),
            "actor_update_rejected": float(ppo_epochs_rejected > 0),
            "return_mean": float(_masked_mean(returns, survival).item()),
            "return_std": float(_masked_variance(returns, survival).sqrt().item()),
            "value_mean_pre": float(_masked_mean(baseline_values, survival).item()),
            "value_mean_post": float(_masked_mean(fitted_values, survival).item()),
            "critic_explained_variance_pre": float(
                critic_explained_variance_pre.item()
            ),
            "critic_explained_variance_post": float(
                critic_explained_variance_post.item()
            ),
            "raw_advantage_mean": float(raw_advantage_mean.item()),
            "raw_advantage_std": float(raw_advantage_variance.sqrt().item()),
            "advantage_mean": float(weighted_advantage_mean.item()),
            "advantage_std": float(weighted_advantage_std.item()),
            "advantage_min": advantage_min,
            "advantage_max": advantage_max,
            "critic_update_count": float(self.critic_update_count),
            "critic_warmup_complete": float(self._critic_warmup_synced),
            "target_hard_sync": target_hard_sync,
            "imagined_reward": float(_masked_mean(
                rollout["imagined_reward"], initial_active,
            ).item()),
            "imagined_samples": float(survival.sum().item()),
            "survival_mass": float(survival.sum().item()),
            "survival_fraction": float(
                survival.sum().item()
                / initial_active.sum().item()
            ),
            "imagined_starts": float(rollout["imagined_starts"].item()),
        }

    def update(self) -> dict[str, float]:
        replay_episodes = len(self.replay)
        replay_frames = self.replay.n_frames
        # Tests and offline callers may populate replay directly, bypassing
        # the online collection boundary.  Keeping replay occupancy as a
        # lower bound supports those callers while online training uses the
        # monotonic counter and therefore cannot be permanently gated by a
        # finite replay capacity.
        collected_frames = max(int(self.collected_frames), int(replay_frames))
        minimum_model_frames = max(
            self.config.min_model_replay_frames,
            self.config.burn_in + 1,
        )
        minimum_agent_frames = max(
            self.config.min_agent_replay_frames,
            minimum_model_frames,
        )
        episode_ready = replay_episodes >= self.config.min_replay_episodes
        model_ready = episode_ready and collected_frames >= minimum_model_frames
        critic_ready = model_ready
        actor_replay_ready = model_ready and collected_frames >= minimum_agent_frames
        critic_warmup_ready = (
            self.critic_update_count >= self.config.critic_warmup_updates
            and self._critic_warmup_synced
        )
        agent_ready = actor_replay_ready and critic_warmup_ready
        readiness = {
            "replay_episode_ready": float(episode_ready),
            "model_ready": float(model_ready),
            "critic_ready": float(critic_ready),
            "actor_replay_ready": float(actor_replay_ready),
            "critic_warmup_ready": float(critic_warmup_ready),
            "agent_ready": float(agent_ready),
            "model_skip_insufficient_replay": float(not model_ready),
            "critic_skip_model_not_ready": float(not critic_ready),
            "agent_skip_insufficient_replay": float(not actor_replay_ready),
            "agent_skip_critic_warmup": float(
                actor_replay_ready and not critic_warmup_ready
            ),
            "agent_skip_model_not_ready": float(not model_ready),
            "min_model_replay_frames": float(minimum_model_frames),
            "min_agent_replay_frames": float(minimum_agent_frames),
            "critic_warmup_updates": float(self.config.critic_warmup_updates),
            "critic_update_count_before": float(self.critic_update_count),
            "replay_episodes": float(replay_episodes),
            "replay_frames": float(replay_frames),
            "collected_frames": float(collected_frames),
        }
        if not model_ready:
            return {
                "skipped": 1.0,
                "model_skipped": 1.0,
                "agent_skipped": 1.0,
                **readiness,
            }
        totals: dict[str, float] = {}
        model_runs = 0
        for _ in range(self.config.model_epochs):
            values = self._train_world_model(self._sample_batch())
            model_runs += 1
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + value
        agent_runs = 0
        if critic_ready:
            for _ in range(self.config.agent_epochs):
                values = self._train_agent(
                    self._sample_batch(), update_actor=agent_ready,
                )
                agent_runs += 1
                for name, value in values.items():
                    totals[name] = totals.get(name, 0.0) + value
        for name in list(totals):
            divisor = agent_runs if name.startswith((
                "agent_", "actor_", "critic_", "policy_", "imagined_",
                "clip_", "survival_", "ppo_", "return_", "value_",
                "advantage_", "raw_advantage_", "target_",
            )) else model_runs
            totals[name] /= max(1, divisor)
        self.update_count += 1
        totals.update({
            "skipped": 0.0,
            "update": float(self.update_count),
            "model_skipped": 0.0,
            **readiness,
        })
        if not agent_ready:
            totals["agent_skipped"] = 1.0
            totals["actor_skipped"] = 1.0
        else:
            totals.setdefault("agent_skipped", 0.0)
            totals.setdefault("actor_skipped", 0.0)
        totals.setdefault("critic_skipped", float(not critic_ready))
        return totals

    def _controller_state_dict(self) -> dict[str, Any]:
        def cpu(state: RSSMState) -> dict[str, Tensor]:
            return {
                "logits": state.logits.detach().cpu(),
                "stoch": state.stoch.detach().cpu(),
                "deter": state.deter.detach().cpu(),
            }
        return {
            "state": cpu(self._controller_state),
            "prior": cpu(self._controller_prior),
            "previous_action": self._controller_previous_action.detach().cpu(),
            "current_presence": self._current_presence.detach().cpu(),
            "previous_presence": self._previous_presence.detach().cpu(),
            "decision_open": self._decision_open,
            "current_episode": None if self._current_episode is None else {
                "episode": self._current_episode.episode,
                "frames": [frame.state_dict() for frame in self._current_episode.frames],
            },
        }

    def state_dict(self) -> dict[str, Any]:
        rng: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng["cuda"] = torch.cuda.get_rng_state_all()
        return {
            "config": self.config.to_dict(),
            "machine_agent_ids": self.machine_agent_ids,
            "world_model": self.world_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "model_optimizer": self.model_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "replay": self.replay.state_dict(),
            "rng": rng,
            "controller": self._controller_state_dict(),
            "update_count": self.update_count,
            "critic_update_count": self.critic_update_count,
            "critic_warmup_synced": self._critic_warmup_synced,
            "episodes_seen": self.episodes_seen,
            "environment_steps": self.environment_steps,
            "collected_frames": self.collected_frames,
        }

    def _load_controller(self, state: Mapping[str, Any]) -> None:
        def rssm(item: Mapping[str, Tensor]) -> RSSMState:
            return RSSMState(
                logits=item["logits"].to(self.device),
                stoch=item["stoch"].to(self.device),
                deter=item["deter"].to(self.device),
            )
        self._controller_state = rssm(state["state"])
        self._controller_prior = rssm(state["prior"])
        self._controller_previous_action = state["previous_action"].to(self.device)
        self._current_presence = state["current_presence"].to(self.device)
        self._previous_presence = state["previous_presence"].to(self.device)
        self._decision_open = bool(state["decision_open"])
        current = state.get("current_episode")
        if current is None:
            self._current_episode = None
        else:
            frames = [
                SparseFrame.from_mapping(
                    frame, obs_dim=self.config.obs_dim, action_dim=self.config.action_dim,
                )
                for frame in current["frames"]
            ]
            self._current_episode = SparseEpisode(int(current["episode"]), frames)

    def load_state_dict(self, state: Mapping[str, Any], *, strict_roles: bool = True) -> None:
        saved_ids = tuple(str(value) for value in state["machine_agent_ids"])
        if strict_roles and saved_ids != self.machine_agent_ids:
            raise ValueError("checkpoint machine-agent role mapping does not match")
        saved_config = MambaConfig.from_dict(state["config"])
        for name in (
            "obs_dim", "action_dim", "n_slots", "hidden_size",
            "model_hidden_size", "embedding_size", "deterministic_size",
            "n_categoricals", "n_classes", "transformer_layers",
            "transformer_heads", "transformer_dropout", "mlp_layers",
        ):
            if getattr(saved_config, name) != getattr(self.config, name):
                raise ValueError(f"checkpoint architecture mismatch for {name}")
        for name in (
            "critic_epochs", "critic_warmup_updates", "imagination_horizon",
            "max_imagination_starts", "imagination_alive_threshold",
            "imagination_reward_scope", "model_lr", "actor_lr", "critic_lr",
            "gamma", "gae_lambda", "ppo_clip", "target_policy_kl",
            "entropy_coef", "value_coef", "policy_max_grad_norm",
            "critic_max_grad_norm", "critic_huber_delta", "target_tau",
            "normalize_advantages", "advantage_clip",
        ):
            if getattr(saved_config, name) != getattr(self.config, name):
                raise ValueError(f"checkpoint training-semantics mismatch for {name}")
        self.world_model.load_state_dict(state["world_model"])
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.model_optimizer.load_state_dict(state["model_optimizer"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.replay.load_state_dict(state["replay"])
        self._load_controller(state["controller"])
        self.update_count = int(state["update_count"])
        self.critic_update_count = int(state.get("critic_update_count", 0))
        self._critic_warmup_synced = bool(state.get(
            "critic_warmup_synced",
            self.critic_update_count >= self.config.critic_warmup_updates,
        ))
        self.episodes_seen = int(state["episodes_seen"])
        self.environment_steps = int(state["environment_steps"])
        self.collected_frames = int(
            state.get("collected_frames", max(self.environment_steps, self.replay.n_frames))
        )
        rng = state["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        # ``torch.load(..., map_location=cuda)`` also relocates serialized RNG
        # byte tensors.  The global CPU generator only accepts a CPU
        # ByteTensor, while CUDA generator restoration likewise expects its
        # state list in host memory.
        torch.set_rng_state(
            torch.as_tensor(rng["torch"], dtype=torch.uint8, device="cpu")
        )
        if "cuda" in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                [
                    torch.as_tensor(item, dtype=torch.uint8, device="cpu")
                    for item in rng["cuda"]
                ]
            )

    def checkpoint_state(
        self,
        episode: int | None = None,
        metrics: Mapping[str, Any] | None = None,
        role_source: str | None = None,
        role_hash: str | None = None,
        config_dict: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model_family": MODEL_FAMILY,
            "episode": self.episodes_seen if episode is None else int(episode),
            "obs_dim": self.config.obs_dim,
            "action_dim": self.config.action_dim,
            "machine_agent_ids": list(self.machine_agent_ids),
            "metrics": dict(metrics or {}),
            "role_source": role_source,
            "role_hash": role_hash,
            "run_config": dict(config_dict or {}),
            "learner": self.state_dict(),
        }

    def load_checkpoint_state(
        self,
        state: Mapping[str, Any],
        *,
        strict_roles: bool = True,
        role_source: str | None = None,
        role_hash: str | None = None,
    ) -> int:
        if int(state.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("unsupported MAMBA checkpoint schema")
        if state.get("model_family") != MODEL_FAMILY:
            raise ValueError("checkpoint model_family is not mamba")
        checkpoint_ids = tuple(str(value) for value in state.get("machine_agent_ids", ()))
        if strict_roles and checkpoint_ids and checkpoint_ids != self.machine_agent_ids:
            raise ValueError("checkpoint machine-agent role mapping does not match")
        if strict_roles and role_source is not None and state.get("role_source") != role_source:
            raise ValueError("checkpoint role_source does not match")
        if strict_roles and role_hash is not None and state.get("role_hash") != role_hash:
            raise ValueError("checkpoint role_hash does not match")
        self.load_state_dict(state["learner"], strict_roles=strict_roles)
        return int(state.get("episode", self.episodes_seen))

    def save_checkpoint(self, path: str | os.PathLike[str], **metadata: Any) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        try:
            torch.save(self.checkpoint_state(**metadata), temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def load_checkpoint(
        self,
        path: str | os.PathLike[str],
        *,
        strict_roles: bool = True,
        role_source: str | None = None,
        role_hash: str | None = None,
    ) -> int:
        state = torch.load(path, map_location=self.device, weights_only=False)
        return self.load_checkpoint_state(
            state,
            strict_roles=strict_roles,
            role_source=role_source,
            role_hash=role_hash,
        )
