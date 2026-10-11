import numpy as np
import pytest
import torch
from types import SimpleNamespace

from tools.evaluate_component_world_models import (
    endpoint_target, evaluate_condition, parser, run, sample_trajectories,
    select_conditions, select_traffic_conditions, summarize,
)
from urban_dreamer.learning.action_events import ActionEvent, pack_action_events
from urban_dreamer.learning.replay import SequenceBatch


def batch_case(frames=4):
    flow = torch.zeros(1, frames, 2, 8)
    flow[..., 1] = .2 + torch.arange(frames)[None, :, None] * .1
    flow[:, :, 0, 2] = .2
    flow[:, 1:, 1, 2] = .1
    snapshots = []
    for step in range(frames):
        snapshots.append(tuple(dict(vehicle_id=name, current_edge=edge, departed=True,
            arrived=False, in_junction=False) for name, edge in
            ([('initial', 0)] if step == 0 else [('initial', 0), ('birth', 1)])))
    windows = [[(ActionEvent('initial', step, step * 30. + 1., 0, 1, 1),
                 ActionEvent('birth', step + 100, step * 30. + 2., 1, 0, 1))
                for step in range(frames - 1)]]
    action = torch.ones(1, frames - 1, 2) / 2
    return SequenceBatch(flow=flow, action=action, departure=torch.zeros_like(action),
        context=torch.zeros(1, frames - 1, 4), active_vehicle_fraction=torch.ones(1, frames - 1),
        vehicles=(snapshots[0],), time=torch.zeros(1), terminal=torch.zeros(1, frames - 1, dtype=torch.bool),
        total_travel_minutes=torch.ones(1), episode_complete=torch.ones(1, dtype=torch.bool),
        episode=torch.zeros(1, dtype=torch.long), start_index=torch.zeros(1, dtype=torch.long),
        total_vehicle_count=torch.tensor([2]), action_events=pack_action_events(windows),
        boundary_vehicles=(tuple(snapshots),), arrival_times=({},))


class Model:
    def __init__(self):
        self.calls = []

    def encode(self, flow):
        return torch.nn.functional.pad(flow, (0, 56))

    def decode(self, z):
        return z[..., :4]

    def transition(self, z, action, *, action_events, **kwargs):
        self.calls.append(action_events)
        result = z.clone()
        result[..., 2] += torch.randn_like(z[..., 2]) * .01
        return result


def test_endpoint_speed_uses_following_boundary_or_nan():
    current = torch.zeros(1, 2, 8)
    current[..., 1] = .2
    current[..., 2] = .3
    following = current.clone()
    following[..., 1] = .8
    limits, caps = torch.tensor([10., 20.]), torch.tensor([5., 10.])
    target = endpoint_target(current, following, caps, limits)
    torch.testing.assert_close(target[..., 0], torch.tensor([[1.5, 3.]]))
    torch.testing.assert_close(target[..., 1], torch.tensor([[5., 10.]]))
    missing = endpoint_target(current, None, caps, limits)
    assert torch.isnan(missing[..., 1]).all()
    assert torch.isfinite(missing[..., 0]).all()


def test_conditions_are_deterministic_group_paired_and_do_not_duplicate():
    entries = [dict(windows=25, demand_group=f'g{i%3}', path=f'p{i}', episode=i) for i in range(9)]
    kwargs = dict(per_group=4, max_groups=2, horizon=17, seed=17)
    first = select_conditions(entries, **kwargs)
    assert first == select_conditions(entries, **kwargs)
    assert len(first) == 8
    assert len({(row['episode'], row['start']) for row in first}) == 8
    assert all(0 <= row['start'] <= 8 for row in first)


def test_recursive_samples_keep_independent_history_and_repeat_events():
    model, batch = Model(), batch_case()
    decoded, risks = sample_trajectories(model, model.encode(batch.flow[:, 0]), batch.action,
        batch.action_events, horizons=[1, 2, 3], samples=64, sample_chunk=8, steps=2,
        capacities=torch.tensor([5., 10.]), limits=torch.tensor([10., 20.]), seed=1)
    assert decoded[3].shape == (64, 1, 2, 2)
    assert risks[3][0] > risks[1][0] * 1.8
    first_events = model.calls[0]
    assert first_events.batch_size == 8 and first_events.time_steps == 1
    assert len(first_events.vehicle_ids) == 16
    assert set(first_events.batch_index.tolist()) == set(range(8))


def test_protocols_keep_cohort_membership_missing_speed_and_risk_separate():
    batch, model = batch_case(), Model()
    rows = evaluate_condition(model, batch, torch.tensor([5., 10.]), torch.tensor([10., 20.]),
        horizons=[1, 2, 3, 8], samples=4, sample_chunk=2, steps=2, device='cpu',
        scales=np.array([1., 10.]), metadata=dict(demand_group='g', episode='p::0'), seed=42)
    assert len(rows) == 12  # three predictors for one_step + three recursive horizons
    one = next(row for row in rows if row['protocol'] == 'one_step_previous_state' and row['predictor'] == 'predictor')
    recursive = [row for row in rows if row['protocol'] == 'recorded_action_recursive_initial_cohort'
                 and row['predictor'] == 'predictor']
    assert one['true_total_count'] == 1.
    assert all(row['true_total_count'] == 1. for row in recursive)
    assert recursive[0]['disagreement_independent_samples']
    assert not recursive[1]['disagreement_independent_samples']
    assert recursive[-1]['speed_mae_all'] is None
    assert recursive[-1]['speed_valid_roads_all'] == 0
    assert one['speed_valid_roads_all'] == 2
    # Main one-step condition retains birth events; fixed initial-cohort strips them.
    assert any('birth' in events.vehicle_ids for events in model.calls)
    assert any(set(events.vehicle_ids) == {'initial'} for events in model.calls)
    result = summarize(rows, seed=3)
    assert len(result) == 12
    assert all(item['clustered']['count_mae_all']['ci_low'] is None for item in result)


def test_fixed_seed_repeats_metrics():
    args = dict(horizons=[1], samples=4, sample_chunk=2, steps=2, device='cpu',
        scales=np.array([1., 10.]), metadata=dict(demand_group='g', episode='p::0'), seed=123)
    values = [evaluate_condition(Model(), batch_case(), torch.tensor([5., 10.]),
              torch.tensor([10., 20.]), **args) for _ in range(2)]
    assert values[0] == values[1]


def test_existing_results_are_never_overwritten(tmp_path):
    (tmp_path / 'states.jsonl').write_text('negative result\n')
    args = parser().parse_args(['--output-dir', str(tmp_path)])
    with pytest.raises(ValueError, match='preserve'):
        run(args)
    assert (tmp_path / 'states.jsonl').read_text() == 'negative result\n'


def test_phase_selection_uses_true_counts_and_keeps_empty_stress_separate():
    flow = np.zeros((8, 2, 8))
    flow[:, 0, 2] = np.array([0., 1., 4., 8., 6., 3., 1., 0.])
    entries = [dict(path='archive', episode=0, windows=7, demand_group='g')]
    data = SimpleNamespace(manifest={'entries': entries}, _episode=lambda *args, **kwargs: {'flow': flow})
    rows = select_traffic_conditions(data, torch.ones(2), per_group=3, max_groups=1, horizon=1, seed=1)
    assert [(row['traffic_phase'], row['start']) for row in rows] == [
        ('early', 2), ('peak', 3), ('drain', 5), ('empty_stress', 0)]
    no_drain = select_traffic_conditions(data, torch.ones(2), per_group=3, max_groups=1, horizon=3, seed=1)
    assert any(row['traffic_phase'] == 'post_peak_no_drain' for row in no_drain)
