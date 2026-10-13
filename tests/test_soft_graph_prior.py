import copy
import io

import pytest
import torch
from torch import nn

from urban_dreamer.models.world_model import (
    ConditionEncoding, Dynamics, LowFrequencyGraphPrior, UrbanDreamerWorldModel,
    independent_transition_fm_loss, recursive_fm_loss,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(127)


def graph():
    # Directed weights, a connected component and an isolated edge.
    return torch.tensor([[0., .25, 0., 0.], [1., 0., .5, 0.],
                         [0., 2., 0., 0.], [0., 0., 0., 0.]])


def laplacian(adjacency):
    symmetric = (adjacency.double() + adjacency.double().T) / 2
    degree = symmetric.sum(-1)
    inv_sqrt = degree.clamp_min(1e-12).rsqrt()
    return (torch.diag((degree > 0).double())
            - inv_sqrt[:, None] * symmetric * inv_sqrt[None, :])


def dynamics(*, action_conditioned=False, filter_input=False):
    result = Dynamics(adjacency=graph(), static_features=torch.ones(4, 4),
        low_frequency_rank=1, prior_filter="soft", prior_filter_strength=3.,
        prior_action_conditioned=action_conditioned, prior_filter_input=filter_input)
    # Retain the actual action encoder and attention, with bounded CPU work.
    result.transformer.layers = nn.ModuleList(result.transformer.layers[:1])
    result.transformer.activation_checkpointing = False
    nn.init.normal_(result.flow_head.weight, std=.02)
    nn.init.normal_(result.graph_prior.output.weight, std=.02)
    return result


def test_soft_filter_preserves_every_mode_with_resolvent_gains():
    prior = LowFrequencyGraphPrior(graph(), rank=1, prior_filter="soft", prior_filter_strength=3.)
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian(graph()))
    actual = prior.project(eigenvectors)
    gains = 1 / (1 + 3 * eigenvalues)
    torch.testing.assert_close(actual, eigenvectors * gains, atol=4e-8, rtol=1e-6)
    assert (gains > 0).all()
    assert actual[:, -1].norm() > .1
    torch.testing.assert_close(actual[:, eigenvalues.abs() < 1e-12],
                               eigenvectors[:, eigenvalues.abs() < 1e-12], atol=1e-7, rtol=1e-6)
    operator = prior.soft_filter_operator
    torch.testing.assert_close(operator, operator.T, rtol=0, atol=1e-7)
    assert (torch.linalg.eigvalsh(operator) > 0).all()
    assert not torch.allclose(operator @ operator, operator)
    # The isolated edge has zero Laplacian and stays untouched.
    torch.testing.assert_close(operator[3], torch.tensor([0., 0., 0., 1.]), atol=0, rtol=0)


def test_strength_zero_is_identity_and_larger_strength_attenuates_nonzero_modes():
    _, vectors = torch.linalg.eigh(laplacian(graph()))
    gains = []
    for strength in (0., 1., 10.):
        prior = LowFrequencyGraphPrior(graph(), prior_filter="soft", prior_filter_strength=strength)
        output = prior.project(vectors)
        gains.append((output * vectors).sum(0))
        if strength == 0:
            torch.testing.assert_close(output, vectors, atol=0, rtol=0)
    assert (gains[0][2:] > gains[1][2:]).all()
    assert (gains[1][2:] > gains[2][2:]).all()


@pytest.mark.parametrize("kwargs,match", [
    ({"prior_filter": "lowpass"}, "prior_filter"),
    ({"prior_filter_strength": -1.}, "prior_filter_strength"),
    ({"prior_filter_strength": float("nan")}, "prior_filter_strength"),
    ({"prior_filter_strength": float("inf")}, "prior_filter_strength"),
    ({"prior_action_conditioned": 0}, "prior_action_conditioned"),
    ({"prior_filter_input": 1}, "prior_filter_input"),
])
def test_prior_configuration_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        LowFrequencyGraphPrior(graph(), **kwargs)


def test_legacy_hard_layout_projection_conditioning_and_noise_are_exact():
    prior = LowFrequencyGraphPrior(graph(), rank=1, noise_scale=.3)
    assert prior.prior_filter == "hard" and prior.prior_action_conditioned and not prior.filter_input
    assert set(prior.state_dict()) == {
        "basis", "state_proj.weight", "state_proj.bias", "upstream_proj.weight",
        "downstream_proj.weight", "condition_proj.weight", "condition_proj.bias",
        "output.weight", "output.bias",
    }
    soft = LowFrequencyGraphPrior(graph(), rank=1, noise_scale=.3, prior_filter="soft")
    soft.load_state_dict(prior.state_dict(), strict=False)
    z = torch.randn(2, 1, 4, 64)
    noise = torch.randn(2, 1, 1, 64)
    condition = ConditionEncoding(torch.randn(2, 1, 4, 256))
    nn.init.normal_(prior.output.weight, std=.03)
    downstream = prior.adjacency / prior.adjacency.sum(-1, keepdim=True).clamp_min(1e-12)
    upstream = prior.adjacency.T / prior.adjacency.T.sum(-1, keepdim=True).clamp_min(1e-12)
    hidden = (prior.state_proj(z) + downstream @ prior.downstream_proj(z)
              + upstream @ prior.upstream_proj(z) + prior.condition_proj(condition.edge_tokens))
    expected = prior.basis @ (prior.basis.T @ prior.output(torch.nn.functional.silu(hidden)))
    torch.testing.assert_close(prior(z, condition), expected, atol=0, rtol=0)
    torch.testing.assert_close(prior.project(z), prior.basis @ (prior.basis.T @ z), atol=0, rtol=0)
    torch.testing.assert_close(prior.sample_noise(z, noise), .3 * (prior.basis @ noise), atol=0, rtol=0)
    torch.testing.assert_close(soft.sample_noise(z, noise), prior.sample_noise(z, noise), atol=0, rtol=0)


def test_soft_state_only_keeps_seeded_legacy_parameter_initialization_and_noise_basis():
    torch.manual_seed(717)
    legacy = LowFrequencyGraphPrior(graph(), rank=1)
    torch.manual_seed(717)
    soft = LowFrequencyGraphPrior(graph(), rank=1, prior_filter="soft", prior_action_conditioned=False)
    for name, tensor in legacy.state_dict().items():
        torch.testing.assert_close(soft.state_dict()[name], tensor, atol=0, rtol=0, msg=name)


def test_state_only_prior_is_action_invariant_but_fm_transition_responds():
    model = dynamics()
    z = torch.randn(1, 1, 4, 64)
    action0, action1 = torch.zeros(1, 1, 4), torch.ones(1, 1, 4)
    noise = torch.randn(1, 1, 1, 64)
    condition0, condition1 = model.encode_condition(action0), model.encode_condition(action1)
    start0, increment0, cached0 = model.prior_start(z, condition0, noise)
    start1, increment1, cached1 = model.prior_start(z, condition1, noise)
    torch.testing.assert_close(increment0, increment1, atol=0, rtol=0)
    torch.testing.assert_close(start0, start1, atol=0, rtol=0)
    torch.testing.assert_close(cached0.reference_tokens, cached1.reference_tokens, atol=0, rtol=0)
    assert not torch.allclose(condition0.edge_tokens, condition1.edge_tokens)
    result0 = model.transition(z, action0, noise=noise, steps=1)
    result1 = model.transition(z, action1, noise=noise, steps=1)
    assert not torch.allclose(result0, result1)


def test_state_only_prior_supervision_does_not_train_action_condition():
    model = dynamics()
    z = torch.randn(1, 1, 4, 64)
    action = torch.randn(1, 1, 4, requires_grad=True)
    condition = model.encode_condition(action)
    prediction = model.graph_prior(z, condition)
    prediction.square().mean().backward()
    assert model.graph_prior.output.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.action_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.edge_action_proj.parameters())
    assert all(not parameter.requires_grad and parameter.grad is None
               for parameter in model.graph_prior.condition_proj.parameters())
    assert action.grad is None


@pytest.mark.parametrize("recursive", [False, True])
@pytest.mark.parametrize("prior_weight", [0., .4])
@pytest.mark.parametrize("filter_input", [False, True])
def test_fm_cannot_train_state_only_prior_but_its_own_loss_can(recursive, prior_weight, filter_input):
    model = dynamics(filter_input=filter_input)
    sequence = torch.randn(1, 3, 4, 64, requires_grad=True)
    action = torch.rand(1, 2, 4)
    kwargs = dict(action=action, prior_weight=prior_weight,
                  noise=torch.zeros(1, 2, 1, 64), flow_time=torch.full((1, 2), .3))
    if recursive:
        loss, _, _ = recursive_fm_loss(model, z_sequence=sequence, steps=1, **kwargs)
    else:
        loss, _, _ = independent_transition_fm_loss(model,
            z_inputs=sequence[:, :2], z_targets=sequence[:, 1:], predict_endpoints=False, **kwargs)
    loss.backward()
    prior_parameters = [parameter for parameter in model.graph_prior.parameters() if parameter.requires_grad]
    if prior_weight == 0:
        assert all(parameter.grad is None for parameter in prior_parameters)
    else:
        assert model.graph_prior.output.weight.grad.abs().sum() > 0
    assert model.action_encoder.edge_conditioning.weight.grad.abs().sum() > 0
    if recursive:
        assert sequence.grad[:, :1].abs().sum() > 0


@pytest.mark.parametrize("filter_input", [False, True])
def test_state_only_source_detach_preserves_direct_current_state_derivative(filter_input):
    model = dynamics(filter_input=filter_input)
    z = torch.randn(1, 1, 4, 64, requires_grad=True)
    start, increment, _ = model.prior_start(z, model.encode_condition(torch.zeros(1, 1, 4)),
                                           torch.zeros(1, 1, 1, 64))
    assert increment.requires_grad
    start.sum().backward()
    torch.testing.assert_close(z.grad, torch.ones_like(z), atol=0, rtol=0)
    assert all(parameter.grad is None for parameter in model.graph_prior.parameters())


def test_disagreement_reuses_transition_source_and_isolates_prior_gradients():
    model = dynamics()
    z, action = torch.randn(1, 4, 64), torch.rand(1, 4)
    noise = torch.randn(2, 1, 1, 64)
    samples, _, _ = model.latent_disagreement(z, action, noise=noise, n_samples=2, steps=1)
    for index in range(2):
        torch.testing.assert_close(samples[index], model.transition(z, action, noise=noise[index], steps=1),
                                   atol=2e-6, rtol=1e-5)
    samples.square().mean().backward()
    assert all(parameter.grad is None for parameter in model.graph_prior.parameters())
    assert model.flow_head.weight.grad.abs().sum() > 0


def test_soft_buffer_checkpoint_roundtrip_and_dtype_device():
    prior = LowFrequencyGraphPrior(graph(), prior_filter="soft", prior_action_conditioned=False)
    assert prior.soft_filter_operator.dtype == torch.float32
    assert "soft_filter_operator" in prior.state_dict()
    stream = io.BytesIO()
    torch.save(prior.state_dict(), stream)
    stream.seek(0)
    restored = copy.deepcopy(prior)
    restored.soft_filter_operator.zero_()
    restored.load_state_dict(torch.load(stream, weights_only=True))
    z = torch.randn(2, 4, 64)
    torch.testing.assert_close(restored.project(z), prior.project(z), atol=0, rtol=0)
    prior = prior.to(device="cpu", dtype=torch.float64)
    actual = prior.project(z.double())
    assert actual.dtype == torch.float64 and actual.device == z.device
    torch.testing.assert_close(actual, prior.soft_filter_operator @ z.double(), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_soft_projection_accepts_input_dtype_without_moving_its_persistent_buffer(dtype):
    prior = LowFrequencyGraphPrior(graph(), prior_filter="soft")
    z = torch.randn(2, 4, 64).to(dtype=dtype)
    result = prior.project(z)
    assert result.dtype == dtype and result.device == z.device
    assert prior.soft_filter_operator.dtype == torch.float32
    torch.testing.assert_close(result, prior.soft_filter_operator.to(dtype=dtype) @ z, atol=0, rtol=0)


@pytest.mark.parametrize("constructor", [LowFrequencyGraphPrior, Dynamics, UrbanDreamerWorldModel])
@pytest.mark.parametrize("filter_input", [False, True])
def test_all_constructors_forward_new_options(constructor, filter_input):
    kwargs = dict(prior_filter="soft", prior_filter_strength=2.5, prior_action_conditioned=False,
                  prior_filter_input=filter_input)
    if constructor is LowFrequencyGraphPrior:
        model = constructor(graph(), **kwargs)
        prior = model
    else:
        model = constructor(adjacency=graph(), static_features=torch.ones(4, 4), **kwargs)
        prior = model.graph_prior if constructor is Dynamics else model.dynamics.graph_prior
    assert prior.prior_filter == "soft"
    assert prior.prior_filter_strength == 2.5
    assert prior.prior_action_conditioned is False
    assert prior.filter_input is filter_input


@pytest.mark.parametrize("kind", ["hard", "soft"])
@pytest.mark.parametrize("filter_input", [False, True])
@pytest.mark.parametrize("action_conditioned", [False, True])
def test_input_filtering_matches_all_state_branches_and_one_output_filter(kind, filter_input, action_conditioned):
    prior = LowFrequencyGraphPrior(graph(), rank=1, prior_filter=kind, prior_filter_strength=2.,
        prior_filter_input=filter_input, prior_action_conditioned=action_conditioned)
    nn.init.normal_(prior.output.weight, std=.02)
    z = torch.randn(2, 1, 4, 64, requires_grad=True)
    condition = ConditionEncoding(torch.randn(2, 1, 4, 256, requires_grad=True))
    input_state = prior.project(z) if filter_input else z
    downstream = prior.adjacency / prior.adjacency.sum(-1, keepdim=True).clamp_min(1e-12)
    upstream = prior.adjacency.T / prior.adjacency.T.sum(-1, keepdim=True).clamp_min(1e-12)
    hidden = (prior.state_proj(input_state) + downstream @ prior.downstream_proj(input_state)
              + upstream @ prior.upstream_proj(input_state))
    if action_conditioned:
        hidden = hidden + prior.condition_proj(condition.edge_tokens)
    expected = prior.project(prior.output(torch.nn.functional.silu(hidden)))
    actual = prior(z, condition)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().mean().backward()
    assert z.grad.abs().sum() > 0
    for branch in (prior.state_proj, prior.upstream_proj, prior.downstream_proj, prior.output):
        assert branch.weight.grad.abs().sum() > 0
    assert (condition.edge_tokens.grad is not None) == action_conditioned


def test_hard_input_filter_removes_high_frequency_information_before_nonlinearity():
    adjacency = torch.tensor([[0., 1.], [1., 0.]])
    prior = LowFrequencyGraphPrior(adjacency, rank=1, prior_action_conditioned=False)
    with torch.no_grad():
        prior.state_proj.weight.copy_(torch.eye(64))
        prior.state_proj.bias.zero_()
        prior.upstream_proj.weight.zero_()
        prior.downstream_proj.weight.zero_()
        prior.output.weight.copy_(torch.eye(64))
    dual = copy.deepcopy(prior)
    dual.filter_input = True
    z = torch.tensor([1., -1.]).view(1, 1, 2, 1).expand(1, 1, 2, 64)
    condition = ConditionEncoding(torch.zeros(1, 1, 2, 256))
    # SiLU mixes spectral modes: an unfiltered high-frequency input can predict
    # a nonzero low-frequency output. Hard input filtering loses that signal.
    assert prior(z, condition).abs().sum() > 1.
    torch.testing.assert_close(dual(z, condition), torch.zeros_like(z), atol=1e-6, rtol=0)
    torch.testing.assert_close(dual.sample_noise(z, torch.ones(1, 1, 1, 64)),
                               prior.sample_noise(z, torch.ones(1, 1, 1, 64)), atol=0, rtol=0)


def test_soft_metadata_reports_full_output_rank_and_separate_noise_rank():
    from urban_dreamer.models.experimental_dynamics import ExperimentalPrior, variant_metadata
    graph = torch.ones(5, 5)-torch.eye(5)
    model = UrbanDreamerWorldModel(adjacency=graph, static_features=torch.zeros(5, 4),
        low_frequency_rank=2, prior_filter='soft', prior_action_conditioned=False)
    metadata = variant_metadata(model)
    assert metadata['noise_rank'] == 2
    assert metadata['projection_rank'] == 5
    assert metadata['background_prior']['action_conditioned'] is False
    wrapped = ExperimentalPrior(model.dynamics.graph_prior, random_projection=False, seed=0, keep_prediction=True)
    assert wrapped.projection_rank == 5
    assert wrapped.prior_action_conditioned is False
    for options in [dict(random_projection=True), dict(random_projection=False, projection_rank=3)]:
        with pytest.raises(ValueError, match='explicit hard'):
            ExperimentalPrior(model.dynamics.graph_prior, seed=0, **options)
