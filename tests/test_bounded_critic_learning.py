"""Predeclared toy calibration, not evidence of real traffic-policy recovery.

Protocol: .omx/plans/critic-learning-validation-protocol.json. Labels are exact
remaining physical returns for one vehicle choosing between two finite paths.
"""
import pytest
import torch
from torch.nn import functional as F

from urban_dreamer.models.event_critic import EventActionCritic
from urban_dreamer.models.policy import (
    CONTEXT_DIM,
    LATENT_DIM,
    CentralizedLatentValueCritic,
)


@pytest.fixture
def deterministic_cpu():
    previous_threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        try:
            yield
        finally:
            torch.set_num_threads(previous_threads)


def _states(x):
    deadline_seconds = 120.0 + 60.0 * x
    budget = deadline_seconds / 60.0
    travel_seconds = deadline_seconds[:, None] * torch.stack(
        (0.25 + 0.1 * x, 0.60 + 0.1 * x), dim=1
    )
    exact_q = -travel_seconds / 60.0
    z = torch.zeros(len(x), 1, LATENT_DIM)
    z[:, 0, 0] = x
    context = torch.zeros(len(x), CONTEXT_DIM)
    context[:, 0] = x
    features = torch.stack(
        (x[:, None].expand(-1, 2), torch.tensor([0.0, 1.0]).expand(len(x), -1)),
        dim=-1,
    )
    return z, context, features, budget, exact_q


@pytest.mark.parametrize("kind", ["v", "q"])
def test_fresh_critics_learn_exact_physical_returns(kind, deterministic_cpu):
    train = _states(torch.linspace(0.0, 1.0, 9))
    heldout = _states((torch.arange(8, dtype=torch.float32) + 0.5) / 8.0)
    factory = CentralizedLatentValueCritic if kind == "v" else lambda **kw: EventActionCritic(2, **kw)
    bounded = factory(bounded=True)
    unbounded = factory(bounded=False)
    # Copy trainable weights only: the bounded contract buffer must stay distinct.
    with torch.no_grad():
        source = dict(bounded.named_parameters())
        for name, parameter in unbounded.named_parameters():
            parameter.copy_(source[name])

    def predict(model, rows, budget=None):
        z, context, features, stored_budget, _ = rows
        budget = stored_budget if budget is None else budget
        if kind == "v":
            return model(z, context, budget=budget).squeeze(-1)
        return model(features, torch.ones(features.shape[:2], dtype=torch.bool), budget=budget)

    def label(rows):
        return rows[-1].mean(-1) if kind == "v" else rows[-1]

    measurements = {}
    for name, model in (("bounded", bounded), ("unbounded", unbounded)):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        for _ in range(350):
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(predict(model, train), label(train))
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            prediction = predict(model, heldout)
            mae = (prediction - label(heldout)).abs().mean().item()
            measurements[name] = mae
            assert mae <= 0.03, (kind, name, "heldout physical-minute MAE", mae)
            if kind == "q":
                clear = (heldout[-1][:, 0] - heldout[-1][:, 1]).abs() >= 0.1
                assert clear.all()
                assert torch.equal(prediction[clear].argmax(-1), heldout[-1][clear].argmax(-1))
    with torch.no_grad():
        terminal = predict(bounded, heldout, budget=torch.zeros(8))
        assert torch.equal(terminal, torch.zeros_like(terminal))
    print(f"toy {kind} heldout physical-minute MAE: {measurements}; real-policy gate NOT evaluated")


def test_physical_return_loss_keeps_budget_squared_gradient_weight(deterministic_cpu):
    model = EventActionCritic(1, bounded=True)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    features = torch.zeros(2, 1, 1)
    budget = torch.tensor([1.0, 2.0])
    q = model(features, torch.ones(2, 1, dtype=torch.bool), budget=budget)
    q.retain_grad()
    # Equal ratio errors remain unequal physical-return errors. Physical MSE
    # therefore weights the raw-head gradient by B squared, not by 1/B squared.
    target = -0.25 * budget[:, None]
    F.mse_loss(q, target).backward()
    assert torch.allclose(q.detach(), -0.5 * budget[:, None])
    assert torch.allclose(q.grad, torch.tensor([[-0.25], [-0.50]]))
    assert model.scorer[-1].bias.grad.item() == pytest.approx(0.3125)
