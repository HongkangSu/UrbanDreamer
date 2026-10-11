import pytest
import torch
from torch.distributions import Categorical

from urban_dreamer.models.policy import NextHopActor


@pytest.fixture
def actor(monkeypatch):
    actor = NextHopActor()
    monkeypatch.setattr(actor, 'forward_logits', lambda *args, **kwargs:
                        torch.tensor([[4., 1., -torch.inf]]))
    return actor


def test_temperature_controls_draws_log_probabilities_and_entropy(actor):
    features, mask = torch.zeros(1, 3, 10), torch.tensor([[True, True, False]])
    torch.manual_seed(73)
    expected = Categorical(logits=torch.tensor([[2., .5, -torch.inf]]))
    draw = expected.sample()
    torch.manual_seed(73)
    selection = actor.sample_action(features, mask, temperature=2.)
    assert torch.equal(selection.candidate_index, draw)
    torch.testing.assert_close(selection.log_probability, expected.log_prob(draw))
    torch.testing.assert_close(selection.entropy, expected.entropy())
    assert selection.candidate_index.item() != 2


def test_training_default_remains_unit_temperature(actor):
    features, mask = torch.zeros(1, 3, 10), torch.tensor([[True, True, False]])
    torch.manual_seed(42)
    default = actor.sample_action(features, mask)
    torch.manual_seed(42)
    explicit = actor.sample_action(features, mask, temperature=1.)
    assert torch.equal(default.candidate_index, explicit.candidate_index)
    torch.testing.assert_close(default.log_probability, explicit.log_probability)
    torch.testing.assert_close(default.entropy, explicit.entropy)


@pytest.mark.parametrize('temperature', [0., -1., float('nan'), float('inf'), True])
def test_invalid_temperatures_are_rejected(actor, temperature):
    with pytest.raises(ValueError, match='temperature'):
        actor.sample_action(torch.zeros(1, 3, 10), torch.tensor([[True, True, False]]),
                            temperature=temperature)


def test_argmax_can_still_be_requested_explicitly(actor):
    selection = actor.sample_action(torch.zeros(1, 3, 10), torch.tensor([[True, True, False]]),
                                   deterministic=True, temperature=2.)
    assert selection.candidate_index.item() == 0
