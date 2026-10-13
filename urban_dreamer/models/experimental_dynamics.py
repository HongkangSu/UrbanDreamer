"""Offline ablations sharing the production dynamics and condition encoder.

Apply a variant before constructing its optimizer and before loading a variant
checkpoint. Production graph-FM is deliberately left completely unchanged.
These adapters change prediction only; cohort and target construction remain the
caller's responsibility. Regression/Gaussian training uses independent windows.
"""
from __future__ import annotations

from dataclasses import replace
import math

import torch
from torch import nn
from torch.nn import functional as F

from urban_dreamer.models.world_model import (
    ConditionEncoding, Dynamics, FLOW_LATENT_DIM, LowFrequencyGraphPrior,
    UrbanDreamerWorldModel, independent_transition_fm_loss,
)

VARIANTS = ("graph_fm", "no_prior_fm", "random_prior_fm", "regression",
            "graph_regression", "gaussian", "graph_prior_r64_fm", "graph_prior_r128_fm",
            "isotropic_noise_graph_fm", "isotropic_noise_no_prior_fm", "isolated_prior_graph_fm")


def _low_frequency_basis(adjacency, rank):
    """Same spectral operator as production, without creating trainable weights."""
    graph = adjacency.detach().float().cpu()
    symmetric = (graph + graph.T) / 2
    degree = symmetric.sum(-1)
    inverse = degree.clamp_min(1e-12).rsqrt()
    laplacian = torch.diag((degree > 0).float()) - inverse[:, None] * symmetric * inverse[None, :]
    selected_rank = min(rank, max(1, len(graph) - 1))
    _, basis = torch.linalg.eigh(laplacian)
    return basis[:, :selected_rank]


class ExperimentalPrior(nn.Module):
    """Separate a fixed source-noise subspace from the coarse-prediction subspace.

    ``rank`` remains the noise coefficient dimension expected by production FM
    losses. ``projection_rank`` records the deterministic projection dimension.
    With isotropic noise, the stored ``noise_basis`` preserves source provenance
    only; sampling uses independent road coordinates with matched total energy.
    """

    def __init__(self, prior: LowFrequencyGraphPrior, *, random_projection: bool, seed: int,
                 projection_rank: int | None = None, isotropic_noise: bool = False,
                 keep_prediction: bool = False):
        super().__init__()
        if projection_rank is not None and (random_projection or isinstance(projection_rank, bool)
                                            or not isinstance(projection_rank, int) or projection_rank < 1):
            raise ValueError("spectral projection rank must be a positive integer without random projection")
        if (random_projection or projection_rank is not None) and getattr(prior, 'prior_filter', 'hard') != 'hard':
            raise ValueError('rank/random projection ablations require an explicit hard prior filter')
        self.original_noise_rank = prior.rank
        self.isotropic_noise = bool(isotropic_noise)
        self.noise_rank = prior.basis.shape[0] if isotropic_noise else prior.rank
        self.noise_scale = prior.noise_scale
        self.register_buffer("noise_basis", prior.basis.detach().clone())
        self.register_buffer("projection_seed", torch.tensor(seed, dtype=torch.long,
                                                           device=prior.basis.device))
        if random_projection:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            basis = torch.linalg.qr(torch.randn(prior.basis.shape, generator=generator),
                                    mode="reduced").Q.to(prior.basis)
            prior.basis.copy_(basis)
            self.prediction = prior
        elif projection_rank is not None:
            prior.basis = _low_frequency_basis(prior.adjacency, projection_rank).to(prior.basis)
            prior.rank = prior.basis.shape[1]
            self.prediction = prior
        elif keep_prediction:
            self.prediction = prior
        else:
            self.prediction = None

    @property
    def rank(self):
        return self.noise_rank

    @property
    def source_std(self):
        """Coordinate std: latent coordinates for isotropic, basis coefficients otherwise."""
        return (self.noise_scale * math.sqrt(self.original_noise_rank / self.noise_rank)
                if self.isotropic_noise else self.noise_scale)

    @property
    def projection_rank(self):
        return 0 if self.prediction is None else getattr(self.prediction, 'projection_rank', self.prediction.basis.shape[1])

    @property
    def prior_action_conditioned(self):
        return True if self.prediction is None else getattr(self.prediction, 'prior_action_conditioned', True)

    @property
    def basis(self):
        return self.noise_basis if self.prediction is None else self.prediction.basis

    def forward(self, z, condition):
        return torch.zeros_like(z) if self.prediction is None else self.prediction(z, condition)

    def project(self, values):
        if self.prediction is None:
            return torch.zeros_like(values)
        return self.prediction.project(values)

    def sample_noise(self, z, noise=None):
        shape = (*z.shape[:-2], self.rank, FLOW_LATENT_DIM)
        if noise is None:
            noise = torch.randn(shape, device=z.device, dtype=z.dtype)
        if tuple(noise.shape) != shape:
            raise ValueError(f"noise must have coefficient shape {shape}")
        if self.isotropic_noise:
            # Keep covariance trace sigma^2 * original_rank per latent channel.
            # No dense identity basis is needed for independent road coordinates.
            return self.source_std * noise.to(z)
        return self.noise_scale * torch.matmul(self.noise_basis.to(z), noise.to(z))


class ExperimentalDynamics(Dynamics):
    """A serializable adapter; all FM computations delegate to the original net."""

    def __init__(self, backbone: Dynamics, variant: str, seed: int):
        nn.Module.__init__(self)
        self.backbone = backbone
        self.variant = variant
        self.seed = int(seed)
        if variant in {"no_prior_fm", "regression", "gaussian", "random_prior_fm"}:
            backbone.graph_prior = ExperimentalPrior(
                backbone.graph_prior, random_projection=variant == "random_prior_fm", seed=seed)
        elif variant in {"graph_prior_r64_fm", "graph_prior_r128_fm"}:
            backbone.graph_prior = ExperimentalPrior(backbone.graph_prior, random_projection=False,
                seed=seed, projection_rank=64 if variant == "graph_prior_r64_fm" else 128)
        elif variant in {"isotropic_noise_graph_fm", "isotropic_noise_no_prior_fm"}:
            backbone.graph_prior = ExperimentalPrior(backbone.graph_prior, random_projection=False,
                seed=seed, isotropic_noise=True, keep_prediction=variant == "isotropic_noise_graph_fm")
        if variant == "gaussian":
            old = backbone.flow_head
            head = nn.Linear(old.in_features, 2 * FLOW_LATENT_DIM,
                             device=old.weight.device, dtype=old.weight.dtype)
            with torch.no_grad():
                head.weight.zero_()
                head.bias.zero_()
                head.weight[:FLOW_LATENT_DIM].copy_(old.weight)
                head.bias[:FLOW_LATENT_DIM].copy_(old.bias)
                # Match the scale of the default FM source perturbation initially.
                head.bias[FLOW_LATENT_DIM:].fill_(math.log(math.expm1(.1)))
            backbone.flow_head = head
        self.train(backbone.training)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("backbone"), name)

    def get_extra_state(self):
        return {"variant": self.variant, "seed": self.seed, "version": 1}

    def set_extra_state(self, state):
        if state["variant"] != self.variant or state.get("version") != 1:
            raise ValueError("apply the checkpoint's variant before loading state_dict")
        self.seed = int(state["seed"])

    def forward(self, *args, **kwargs):
        return self.backbone(*args, **kwargs)

    def distribution(self, z, action, *, action_events=None):
        """Return next-latent mean, optional diagonal std, and coarse increment."""
        if z.ndim != 4 or z.shape[-2:] != (self.n_edges, FLOW_LATENT_DIM):
            raise ValueError("distribution expects z with shape (B,T,E,64)")
        condition = self.encode_condition(action, action_events=action_events)
        increment = self.graph_prior(z, condition)
        coarse = z + increment
        condition = ConditionEncoding(condition.edge_tokens, self.reference_proj(coarse))
        raw = self.backbone(action, packed_latents=coarse,
                            flow_time=z.new_zeros(z.shape[:2]), condition=condition).latent_velocity
        if self.variant == "gaussian":
            delta, raw_scale = raw.chunk(2, dim=-1)
            return coarse + delta, F.softplus(raw_scale.float()) + 1e-4, increment
        return coarse + raw, None, increment

    def transition(self, z, action, *, noise=None, steps=2, solver="euler",
                   picard_iterations=None, picard_batch_chunk_size=64, action_events=None):
        if self.variant.endswith("fm"):
            return super().transition(z, action, noise=noise, steps=steps, solver=solver,
                picard_iterations=picard_iterations, picard_batch_chunk_size=picard_batch_chunk_size,
                action_events=action_events)
        squeezed = z.ndim == 3
        if squeezed:
            z, action = z.unsqueeze(1), action.unsqueeze(1)
            if noise is not None:
                noise = noise.unsqueeze(1)
        mean, std, _ = self.distribution(z, action, action_events=action_events)
        if std is not None:
            if noise is None:
                noise = torch.randn_like(mean)
            if noise.shape != mean.shape:
                raise ValueError(f"Gaussian noise must have full latent shape {tuple(mean.shape)}")
            mean = mean + std.to(mean) * noise.to(mean)
        result = mean.squeeze(1) if squeezed else mean
        return result

    def latent_disagreement(self, z, action, *, n_samples=2, noise=None, steps=2,
                            solver="euler", picard_iterations=None,
                            picard_batch_chunk_size=64, action_events=None):
        if self.variant.endswith("fm"):
            return super().latent_disagreement(z, action, n_samples=n_samples, noise=noise,
                steps=steps, solver=solver, picard_iterations=picard_iterations,
                picard_batch_chunk_size=picard_batch_chunk_size, action_events=action_events)
        if n_samples < 2 or int(n_samples) != n_samples:
            raise ValueError("latent disagreement requires at least two samples")
        squeezed = z.ndim == 3
        mean, std, _ = self.distribution(z.unsqueeze(1) if squeezed else z,
            action.unsqueeze(1) if squeezed else action, action_events=action_events)
        if squeezed:
            mean = mean.squeeze(1)
            std = None if std is None else std.squeeze(1)
        samples = mean.unsqueeze(0).expand(n_samples, *mean.shape)
        if std is not None:
            if noise is None:
                noise = torch.randn_like(samples)
            if noise.shape != samples.shape:
                raise ValueError(f"Gaussian sample noise must have shape {tuple(samples.shape)}")
            samples = samples + std.to(samples).unsqueeze(0) * noise.to(samples)
        edge_variance = samples.float().var(dim=0, unbiased=True).mean(dim=-1)
        return samples, edge_variance.mean(dim=-1), edge_variance


def apply_variant(world_model: UrbanDreamerWorldModel, variant: str, seed: int = 0):
    """Mutate one fresh world model, then return it. Call before optimizer creation."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    if isinstance(world_model.dynamics, ExperimentalDynamics):
        raise ValueError("apply_variant requires a fresh production world model")
    if variant != "graph_fm":
        world_model.dynamics = ExperimentalDynamics(world_model.dynamics, variant, seed)
    return world_model


def _dynamics(model):
    return model.dynamics if isinstance(model, UrbanDreamerWorldModel) else model


def variant_metadata(model):
    dynamics = _dynamics(model)
    if getattr(dynamics, 'variant', None) == 'residual_fm_regression_center_v1':
        return dynamics.metadata()
    prior = dynamics.graph_prior
    variant = getattr(dynamics, "variant", "graph_fm")
    original_noise_rank = getattr(prior, "original_noise_rank", prior.rank)
    projection_rank = getattr(prior, "projection_rank", prior.rank)
    metadata = {"variant": variant,
            "projection_seed": getattr(dynamics, "seed", None),
            "parameters": sum(p.numel() for p in dynamics.parameters()),
            "trainable_parameters": sum(p.numel() for p in dynamics.parameters() if p.requires_grad),
            "noise_rank": getattr(prior, "noise_rank", prior.rank),
            "original_noise_rank": original_noise_rank,
            "projection_rank": projection_rank,
            "coarse_projection_rank": projection_rank,
            "source_std": getattr(prior, "source_std", prior.noise_scale),
            "source_std_coordinates": "independent_road_latent" if getattr(prior, "isotropic_noise", False) else "spectral_basis_coefficients",
            "expected_energy": prior.noise_scale ** 2 * original_noise_rank * FLOW_LATENT_DIM,
            "expected_energy_unit": "sum of squared source perturbations over roads and all latent channels per state",
            "gaussian_noise_space": "full_latent" if getattr(dynamics, "variant", None) == "gaussian" else None}
    prediction = getattr(prior, 'prediction', prior)
    if prediction is not None and (getattr(prediction, 'prior_filter', 'hard') != 'hard'
            or not getattr(prediction, 'prior_action_conditioned', True)
            or getattr(prediction, 'filter_input', False)):
        metadata['background_prior'] = dict(
            filter=getattr(prediction, 'prior_filter', 'hard'),
            strength=getattr(prediction, 'prior_filter_strength', None),
            input_filtered=getattr(prediction, 'filter_input', False),
            action_conditioned=getattr(prediction, 'prior_action_conditioned', True))
    if not variant.endswith("fm"):
        # The unused prior noise helper is not the predictor's sampling law.
        gaussian = variant == "gaussian"
        metadata.update(noise_rank=dynamics.n_edges if gaussian else 0,
            original_noise_rank=None, source_std=None if gaussian else 0.,
            source_std_coordinates="learned_diagonal_latent" if gaussian else "deterministic",
            expected_energy=None if gaussian else 0.,
            expected_energy_unit="input-dependent sum of predicted latent variances" if gaussian
                                 else "deterministic predictor has no sampled perturbation")
    if variant == "isolated_prior_graph_fm":
        metadata["gradient_policy"] = dict(
            scope="independent_transition_fm_loss_only",
            fm_prior_increment="detached_in_source_and_coarse_reference",
            prior_supervision="original_increment_trainable",
            condition_encoder="shared_trainable; representation is not frozen",
            reference_projection="trainable",
            current_state="direct derivative preserved in prior_start; independent observed inputs remain detached",
            inference="unchanged")
    return metadata


def predict_next(model, z, action, **kwargs):
    return _dynamics(model).transition(z, action, **kwargs)


def experimental_loss(model, *, z_inputs, z_targets, action, action_events=None,
                      prior_weight=.1, predict_endpoints=True, steps=2, solver="euler",
                      picard_iterations=None, noise=None, flow_time=None):
    """Independent-window latent loss; returned endpoints are detached diagnostics."""
    dynamics = _dynamics(model)
    variant = getattr(dynamics, "variant", "graph_fm")
    if variant.endswith("fm"):
        return independent_transition_fm_loss(dynamics, z_inputs=z_inputs, z_targets=z_targets,
            action=action, action_events=action_events,
            prior_weight=0. if variant in {"no_prior_fm", "isotropic_noise_no_prior_fm"} else prior_weight,
            predict_endpoints=predict_endpoints, steps=steps, solver=solver,
            picard_iterations=picard_iterations, noise=noise, flow_time=flow_time,
            detach_prior_increment=variant == "isolated_prior_graph_fm")
    if (z_inputs.ndim != 4 or z_targets.shape != z_inputs.shape
            or z_inputs.shape[0] < 1 or z_inputs.shape[1] < 1):
        raise ValueError("z_inputs and z_targets must have matching nonempty (B,H,E,64) shapes")
    if not math.isfinite(prior_weight) or prior_weight < 0:
        raise ValueError("prior_weight must be finite and nonnegative")
    batch, horizon, edges, width = z_inputs.shape
    if action.shape != z_inputs.shape[:-1]:
        raise ValueError("action must have shape (B,H,E)")
    if action_events is not None:
        if action_events.batch_size != batch or action_events.time_steps < horizon:
            raise ValueError("action events must cover all independent windows")
        selected = action_events.select(action_events.time_index < horizon)
        action_events = replace(selected, batch_index=selected.batch_index * horizon + selected.time_index,
            time_index=torch.zeros_like(selected.time_index), batch_size=batch * horizon, time_steps=1)
    current = z_inputs.detach().reshape(batch * horizon, 1, edges, width)
    target = z_targets.detach().reshape_as(current)
    mean, std, increment = dynamics.distribution(current, action.reshape(batch * horizon, 1, edges),
                                                action_events=action_events)
    mse = (mean.float() - target.float()).square().mean()
    if std is None:
        loss = mse
        metrics = {"regression_mse": mse.detach()}
    else:
        loss = (.5 * ((target.float() - mean.float()) / std).square()
                + std.log() + .5 * math.log(2 * math.pi)).mean()
        metrics = {"gaussian_nll": loss.detach(), "gaussian_std": std.mean().detach(),
                   "regression_mse": mse.detach()}
    prior_mse = (increment.float() - dynamics.graph_prior.project(target - current).detach().float()).square().mean()
    loss = loss + prior_weight * prior_mse
    metrics.update(prior_mse=prior_mse.detach(), prior_loss=(prior_weight * prior_mse).detach())
    endpoint = mean if predict_endpoints else target
    return loss, metrics, endpoint.detach().reshape_as(z_inputs)
