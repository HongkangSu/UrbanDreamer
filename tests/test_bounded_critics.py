"""Output units, physical budgets and explicit critic checkpoint contracts."""
import pytest
import torch

from urban_dreamer.learning.critic_budget import remaining_cost_budget
from urban_dreamer.models.policy import CentralizedLatentValueCritic, CONTEXT_DIM, LATENT_DIM
from urban_dreamer.models.event_critic import EventActionCritic
from urban_dreamer.workflow.config import ConfigError, load_config
from test_urban_dreamer_config import _write_config


def test_budget_units_empty_terminal_and_arrival_invariant():
    assert remaining_cost_budget(60, 180, 3, 6) == 1.
    assert remaining_cost_budget(60, 180, 0, 0) == 0.
    assert remaining_cost_budget(180, 180, 3, 6) == 0.
    assert remaining_cost_budget(180 + 5e-8, 180, 3, 6) == 0.
    # Three vehicles accrue 30 seconds, then one arrives, leaving two.
    cost = 3 * 30 / (60 * 6)
    assert cost + remaining_cost_budget(90, 180, 2, 6) <= remaining_cost_budget(60, 180, 3, 6)


@pytest.mark.parametrize('args', [(0, 10, 1, 0), (0, 10, 2, 1), (0, 10, -1, 2),
    (0, 10, 1.5, 2), (0, 10, 1, 2.5), (0, 10, True, 2), (-1, 10, 1, 2),
    (11, 10, 1, 2), (0, float('inf'), 1, 2), (float('nan'), 10, 1, 2)])
def test_invalid_budget_rejected(args):
    with pytest.raises(ValueError):
        remaining_cost_budget(*args)


def make_case(kind, bounded=True):
    if kind == 'V':
        model = CentralizedLatentValueCritic(bounded=bounded)
        args = (torch.randn(3, 4, LATENT_DIM), torch.randn(3, CONTEXT_DIM))
    else:
        model = EventActionCritic(input_dim=5, bounded=bounded)
        features = torch.randn(3, 2, 5, requires_grad=True)
        mask = torch.tensor([[True, False], [True, True], [True, True]])
        args = (features, mask)
    return model, args


@pytest.mark.parametrize('kind', ['V', 'Q'])
@pytest.mark.parametrize('autocast', [False, True])
def test_outputs_are_physical_returns_with_finite_gradients(kind, autocast):
    torch.manual_seed(5)
    model, args = make_case(kind)
    budget = torch.tensor([0., 1e-9, 4.], dtype=torch.float32)
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=autocast):
        result = model(*args, budget=budget)
    assert result.dtype == torch.float32
    assert result.shape == (3, 1 if kind == 'V' else 2)
    assert torch.isfinite(result).all()
    assert (result >= -budget[:, None]).all() and (result <= 0).all()
    assert torch.equal(result[0], torch.zeros_like(result[0]))
    assert result[-1].abs().min() > .1  # Return units, not raw unitless fractions.
    result.square().mean().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients) > 0
    if kind == 'Q':
        assert args[0].grad is None
        assert result[0, 1] == 0


@pytest.mark.parametrize('kind', ['V', 'Q'])
@pytest.mark.parametrize('budget', [None, torch.tensor(1.), torch.ones(3, 1),
    torch.ones(3, dtype=torch.float64), torch.ones(3, requires_grad=True),
    torch.tensor([1., -1., 1.]), torch.tensor([1., float('nan'), 1.])])
def test_bounded_requires_valid_detached_fp32_vector(kind, budget):
    model, args = make_case(kind)
    with pytest.raises(ValueError):
        model(*args, budget=budget)


@pytest.mark.parametrize('kind', ['V', 'Q'])
def test_contract_rejects_old_weights_even_with_non_strict_load(kind):
    legacy, args = make_case(kind, bounded=False)
    bounded, _ = make_case(kind)
    assert not any('contract' in key for key in legacy.state_dict())
    for strict in (True, False):
        with pytest.raises(RuntimeError, match='explicit migration'):
            bounded.load_state_dict(legacy.state_dict(), strict=strict)
    bounded.load_state_dict(bounded.state_dict())
    restored, _ = make_case(kind, bounded=False)
    restored.load_state_dict(legacy.state_dict())
    assert torch.equal(legacy(*args), restored(*args))


@pytest.mark.parametrize('kind', ['V', 'Q'])
def test_zero_budget_has_exactly_zero_parameter_gradients(kind):
    model, args = make_case(kind)
    model(*args, budget=torch.zeros(3)).sum().backward()
    assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in model.parameters())


def test_masked_q_zero_at_nonzero_budget():
    model, args = make_case('Q')
    result = model(*args, budget=torch.ones(3))
    assert result[0, 1] == 0
    assert result[0, 0] < 0


def test_legacy_fingerprint_preserved_and_bounded_settings_identified(tmp_path, monkeypatch):
    import urban_dreamer.workflow.config as module
    path = _write_config(tmp_path / 'config.toml')
    default = load_config(path)
    original = module.asdict
    def old_asdict(value):
        result = original(value)
        if isinstance(value, module.MethodConfig):
            result = {k: v for k, v in result.items() if not k.startswith('critic_')}
        return result
    with monkeypatch.context() as patch:
        patch.setattr(module, 'asdict', old_asdict)
        previous = module._semantic_fingerprint(**{key: getattr(default, key) for key in
            ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')})
    assert default.semantic_fingerprint == previous
    base = path.read_text()
    path.write_text(base + '\n[method]\ncritic_value_mode = "bounded_cohort_v1"\n')
    enabled = load_config(path)
    assert enabled.semantic_fingerprint != previous
    for field, value in [('critic_target_tau', '0.1'), ('critic_terminal_aux_ratio', '0.5')]:
        path.write_text(base + '\n[method]\ncritic_value_mode = "bounded_cohort_v1"\n' + field + ' = ' + value)
        assert load_config(path).semantic_fingerprint != enabled.semantic_fingerprint


@pytest.mark.parametrize('setting', ['critic_value_mode = "unknown"', 'critic_target_tau = 0',
    'critic_target_tau = 1.1', 'critic_target_tau = nan', 'critic_target_tau = true',
    'critic_terminal_aux_ratio = -1', 'critic_terminal_aux_ratio = inf',
    'critic_value_mode = "bounded_cohort_v1"\ndisagreement_weight = 0.1'])
def test_critic_config_rejects_invalid_settings(tmp_path, setting):
    with pytest.raises(ConfigError):
        load_config(_write_config(tmp_path / 'config.toml', extra='\n[method]\n' + setting))


@pytest.mark.parametrize('kind', ['V', 'Q'])
def test_zero_raw_head_converts_half_fraction_to_budget_scaled_return(kind):
    model, args = make_case(kind)
    head = model.value_head if kind == 'V' else model.scorer[-1]
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    budgets = torch.tensor([2., 10., 40.])
    result = model(*args, budget=budgets)
    expected = -budgets[:, None].expand_as(result) / 2
    if kind == 'Q':
        expected = expected.masked_fill(~args[1], 0.)
    assert torch.equal(result, expected)


@pytest.mark.parametrize('kind', ['V', 'Q'])
def test_bounded_contract_version_must_match(kind):
    model, _ = make_case(kind)
    state = model.state_dict()
    state['_bounded_cohort_contract'] = torch.tensor(2)
    with pytest.raises(RuntimeError, match='explicit migration'):
        model.load_state_dict(state, strict=False)


def test_q_padding_can_contain_nonfinite_values_without_affecting_legal_q():
    model, args = make_case('Q')
    features, mask = args
    with torch.no_grad():
        features[~mask] = float('nan')
    result = model(features, mask, budget=torch.ones(3))
    assert torch.isfinite(result).all()
    assert (result[~mask] == 0).all()
