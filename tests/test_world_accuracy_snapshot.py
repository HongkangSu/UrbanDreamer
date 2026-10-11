import pytest
import torch
from types import SimpleNamespace
from torch import nn

from tools.evaluate_world_accuracy_snapshot import road_scores, fixed_cohort_rollout


def test_empty_roads_are_unavailable_but_ghost_count_is_measured():
    target = torch.zeros(1, 2, 8)
    prediction = target.clone()
    prediction[..., 2] = 1
    scores = road_scores(prediction, target, target[..., 1], torch.tensor([2., 3.]), torch.ones(2))
    assert scores['occupied_speed_mae_mps'] is None
    assert scores['slow_speed_mae_mps'] is None
    assert scores['total_count_true'] == 0
    assert scores['total_count_bias_vehicles'] == 5


def test_occupied_and_slow_scores_use_physical_counts_and_speed():
    target = torch.zeros(1, 2, 8)
    target[0, :, 2] = torch.tensor([.5, .1])
    speed = torch.tensor([[.1, .5]])
    prediction = target.clone()
    prediction[..., 1] = .3
    scores = road_scores(prediction, target, speed, torch.tensor([10., 10.]), torch.tensor([10., 10.]))
    assert scores['occupied_roads'] == 2
    assert scores['slow_roads'] == 1
    assert scores['slow_speed_mae_mps'] == pytest.approx(2.)
    assert scores['occupied_speed_mae_mps'] == pytest.approx(2.)


def test_nonfinite_predictions_are_rejected():
    target = torch.zeros(1, 2, 8)
    prediction = target.clone()
    prediction[..., 1] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        road_scores(prediction, target, target[..., 1], torch.ones(2), torch.ones(2))


def test_rollout_reuses_sample_zero_without_feeding_future_truth(monkeypatch):
    from urban_dreamer.learning.action_events import pack_action_events
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = []
        def encode(self, flow):
            return flow.clone()
        def decode(self, z):
            return z.clone()
        def transition(self, z, action, **kwargs):
            self.inputs.append(z.clone())
            output = z.clone()
            output[..., 1] += torch.arange(1, 4)[:, None]
            return output
    flow = torch.zeros(1, 10, 2, 8)
    flow[:, 1:, :, 1] = 100.
    boundary = [{'vehicle_id': 'car', 'departed': True, 'arrived': False}]
    scoped = SimpleNamespace(transition_flow=flow,
        transition_action=torch.zeros(1, 9, 2),
        transition_action_events=pack_action_events([[() for _ in range(9)]]),
        boundary_vehicles=[[boundary for _ in range(10)]])
    monkeypatch.setattr('tools.evaluate_world_accuracy_snapshot.prepare_cohort_batch', lambda batch, **kw: scoped)
    model = Model()
    trainer = SimpleNamespace(world_model=model, device=torch.device('cpu'),
        config=SimpleNamespace(method=SimpleNamespace(fm_steps=100, fm_solver='euler', picard_iterations=2)),
        network=SimpleNamespace(capacities=[1., 1.], speed_limits_mps=[1., 1.]))
    rows = fixed_cohort_rollout(trainer, scoped, seed=1)
    assert len(model.inputs) == 8
    for step, inputs in enumerate(model.inputs):
        assert torch.equal(inputs[..., 1], torch.full((3, 2), float(step)))
    assert rows[1]['sample0']['speed_mae_mps'] == 97.
    assert rows[1]['mean3']['speed_mae_mps'] == 96.
