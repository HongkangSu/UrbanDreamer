"""Global event returns preserve exact physical cost and endpoint risk accounting."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from urban_dreamer.learning.training import MainlineTrainer


class Candidates:
    def __init__(self, lengths):
        self.network = SimpleNamespace(edge_count=3, lengths_m=np.array(lengths),
                                       speed_limits_mps=np.ones(3))

    def candidates(self, current, destination):
        return (current + 1,) if current < destination else ()

    def junction(self, current, successor):
        return 0., 1.


class Critic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.))

    def forward(self, latent, context):
        return self.bias + context[:, :1]


class Actor:
    def __init__(self):
        self.logit = torch.nn.Parameter(torch.tensor(0.))

    def sample_action(self, features, mask, deterministic=False):
        count = len(features)
        return SimpleNamespace(candidate_index=torch.zeros(count, dtype=torch.long),
            log_probability=self.logit.expand(count), entropy=torch.zeros(count))


def seed(vehicle_id='v', **updates):
    state = dict(vehicle_id=vehicle_id, current_edge=0, destination_edge=2,
        departure_time=0., departed=True, arrived=False, remaining_distance=10.,
        committed_next_edge=None, pending_decision=True, in_junction=False,
        junction_remaining_distance=0., junction_speed=1.)
    return dict(state, **updates)


def make_case(*, deadline=30., lengths=(10., 10., 10.), states=None, population=1):
    trainer = MainlineTrainer.__new__(MainlineTrainer)
    trainer.device = torch.device('cpu')
    trainer.candidates = Candidates(lengths)
    trainer.network = trainer.candidates.network
    trainer.microstep_seconds = .001
    trainer.config = SimpleNamespace(method=SimpleNamespace(disagreement_weight=.25),
        simulation=SimpleNamespace(vehicle_max_speed_mps=10.))
    trainer._episode_end_time = lambda: deadline
    trainer._tensor = lambda value: torch.as_tensor(value, dtype=torch.float32)
    trainer._context = lambda states, timestamp, total_vehicle_count: np.array([
        timestamp / 300., sum(s['departed'] and not s['arrived'] for s in states) / total_vehicle_count])
    trainer._observations = lambda events, flow, context, now: SimpleNamespace(
        features=np.zeros((len(events), 1, 10)), mask=np.ones((len(events), 1), dtype=bool),
        edge_indices=np.array([[s['current_edge'] + 1] for s in events]))
    trainer.actor = Actor()
    trainer.latent_critic = Critic()
    def decode(z):
        output = torch.zeros(1, 3, 4)
        output[..., 1] = 1.
        return output
    trainer.world_model = SimpleNamespace(encode=lambda flow: torch.zeros(1, 3, 1), decode=decode)
    calls = []
    def predict(z, action):
        calls.append(action.copy())
        return z + .1, torch.tensor(2.)
    trainer._predict_with_disagreement = predict
    batch = SimpleNamespace(vehicles=(tuple(states or [seed(remaining_distance=lengths[0])]),),
        time=torch.tensor([0.]), total_vehicle_count=torch.tensor([population]),
        flow=torch.zeros(1, 1, 3, 8))
    return trainer, batch, calls


def test_late_and_simultaneous_actions_share_global_suffix_and_full_endpoint_risk():
    trainer, batch, calls = make_case(states=[seed('a'), seed('b')], population=4)
    rows, loss = trainer._imagine_from_seed(batch, 0, rollout_index=0)
    assert len(calls) == 1
    assert [row['decision_time'] for row in rows] == [0., 0., 10., 10.]
    # Original N=4 is retained although this seed contains only two vehicles.
    assert [row['global_return'].item() for row in rows] == pytest.approx([-.75, -.75, -2/3, -2/3])
    assert rows[0]['global_advantage'].item() == pytest.approx(-.75)
    assert rows[2]['global_advantage'].item() == pytest.approx(-2/3 - 10/300)
    assert rows[0]['global_advantage'] == rows[1]['global_advantage']
    assert rows[2]['global_advantage'] == rows[3]['global_advantage']
    assert all(row['global_reward'].item() == pytest.approx(-.75) for row in rows)
    loss.backward()
    assert trainer.latent_critic.bias.grad is not None
    assert trainer.actor.logit.grad is None


def test_empty_event_windows_still_pay_one_risk_each(tmp_path, monkeypatch):
    import urban_dreamer.learning.training as module
    trainer, batch, calls = make_case(deadline=90., population=2,
        states=[seed(remaining_distance=1e6, pending_decision=False, committed_next_edge=1)])
    captured = []
    original = module.lambda_return
    def capture(rewards, next_values, *, terminal):
        captured.append((rewards.clone(), terminal.clone()))
        return original(rewards, next_values, terminal=terminal)
    monkeypatch.setattr(module, 'lambda_return', capture)
    rows, loss = trainer._imagine_from_seed(batch, 0, rollout_index=0)
    assert rows == [] and len(calls) == 3
    rewards, terminal = captured[0]
    torch.testing.assert_close(rewards, torch.tensor([-.75, -.75, -.75]))
    assert terminal.tolist() == [False, False, True]
    assert torch.isfinite(loss)


def test_partial_deadline_tail_uses_exact_cost_without_model_or_risk():
    trainer, batch, calls = make_case(deadline=25., population=2,
        states=[seed(remaining_distance=1e6)])
    rows, _ = trainer._imagine_from_seed(batch, 0, rollout_index=0)
    assert calls == []
    assert len(rows) == 1
    assert rows[0]['global_return'].item() == pytest.approx(-25/120)
    assert rows[0]['timeout_return'] == rows[0]['timeout_penalty'] == -5000.


def test_midwindow_cohort_arrival_terminates_without_fictitious_full_transition():
    trainer, batch, calls = make_case(lengths=(2., 3., 5.))
    rows, _ = trainer._imagine_from_seed(batch, 0, rollout_index=0)
    assert calls == []
    assert [row['decision_time'] for row in rows] == [0., 2.]
    assert [row['global_return'].item() for row in rows] == pytest.approx([-10/60, -8/60])
    assert all(row['timeout_return'] == row['timeout_penalty'] == 0. for row in rows)


def test_truncation_bootstraps_full_suffix_and_does_not_penalize_unfinished():
    trainer, batch, calls = make_case(deadline=300., population=2,
        states=[seed(remaining_distance=1e6)])
    rows, _ = trainer._imagine_from_seed(batch, 0, rollout_index=0)
    assert len(calls) == 8 and len(rows) == 1
    assert rows[0]['global_return'].item() == pytest.approx(-8*.75 + 240/300)
    assert rows[0]['timeout_return'] == rows[0]['timeout_penalty'] == 0.


def test_single_cohort_arrival_does_not_terminate_other_vehicles():
    trainer, batch, calls = make_case(deadline=60., population=3, states=[
        seed('fast', current_edge=2, destination_edge=2, remaining_distance=5., pending_decision=False),
        seed('slow', remaining_distance=1e6)])
    rows, _ = trainer._imagine_from_seed(batch, 0, rollout_index=0)
    assert len(calls) == 2 and len(rows) == 1
    assert rows[0]['global_return'].item() == pytest.approx(-65/180 - 1.)
    assert rows[0]['timeout_return'] == rows[0]['timeout_penalty'] == -5000.


def test_undeparted_future_seed_is_excluded_from_cost_actions_and_terminal_penalty():
    trainer, batch, calls = make_case(deadline=25., population=7, states=[
        seed('active', remaining_distance=1e6),
        seed('future', departed=False, departure_time=1., pending_decision=False)])
    rows, _ = trainer._imagine_from_seed(batch, 0, rollout_index=0)
    assert not calls and len(rows) == 1
    assert rows[0]['global_return'].item() == pytest.approx(-25/(7*60))
    assert rows[0]['vehicle_id'] == 'active'
    assert rows[0]['timeout_return'] == rows[0]['timeout_penalty'] == -5000.


def test_timeout_is_individual_ledger_once_and_does_not_leak_between_rollouts():
    trainer, batch, _ = make_case(deadline=30., lengths=(2., 3., 5.), population=2,
        states=[seed('arrived', remaining_distance=2.), seed('unfinished', remaining_distance=27.)])
    rows, loss = trainer._imagine_from_seed(batch, 0, rollout_index=0)
    first = [row for row in rows if row['decision_time'] == 0.]
    assert len(first) == 2
    # Both simultaneous actions see the same physical global return and baseline.
    torch.testing.assert_close(first[0]['global_return'], first[1]['global_return'])
    torch.testing.assert_close(first[0]['global_advantage'], first[1]['global_advantage'])
    successful = [row for row in rows if row['vehicle_id'] == 'arrived']
    failed = [row for row in rows if row['vehicle_id'] == 'unfinished']
    assert len(successful) == len(failed) == 2
    assert all(row['timeout_return'] == row['timeout_penalty'] == 0. for row in successful)
    assert [row['timeout_return'] for row in failed] == [-5000., -5000.]
    assert [row['timeout_penalty'] for row in failed] == [0., -5000.]
    assert sum(row['timeout_penalty'] for row in rows) == -5000.
    assert torch.isfinite(loss)
    batch.vehicles[0][1]['remaining_distance'] = 2.
    next_rows, _ = trainer._imagine_from_seed(batch, 0, rollout_index=1)
    assert all(row['timeout_return'] == row['timeout_penalty'] == 0. for row in next_rows)


def test_arrival_exactly_at_deadline_has_no_individual_penalty():
    trainer, batch, _ = make_case(deadline=30.)
    rows, _ = trainer._imagine_from_seed(batch, 0, rollout_index=0)
    assert len(rows) == 2
    assert all(row['timeout_return'] == row['timeout_penalty'] == 0. for row in rows)
