"""Branch restoration admission and causal finite-action decision fidelity."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools import evaluate_pretrained_decision_fidelity as trial
from test_city_imagination import table
from test_event_imagination_protocol import network, seed
from urban_dreamer.learning.global_reward import RewardAccounting
from urban_dreamer.traffic.environment import NextHopDecision
from urban_dreamer.traffic.topology import NextHopCandidates


def test_complete_pretraining_and_zero_online_are_mandatory(tmp_path, monkeypatch):
    folder = tmp_path/'snapshot/checkpoints'
    folder.mkdir(parents=True)
    path = folder/'latest.pt'
    payload = dict(phase='online_training', next_unit=0, training_state=dict(
        metrics=dict(world_model_initialization_updates=49636, online_episodes_completed=0),
        next_environment_episode=0, replay={'online': {'episodes': []}}))
    torch.save(payload, path)
    config = tmp_path/'snapshot/run_config.toml'
    config.write_text('frozen config')
    def parent():
        (tmp_path/'protocol.json').write_text(json.dumps(dict(checkpoint_sha256=trial.sha256(path),
            config_sha256=trial.sha256(config))))
    parent()
    monkeypatch.setattr(trial, 'PRETRAINED_SHA256', trial.sha256(path))
    assert trial.validate_pretrained(tmp_path)['initialization_updates'] == 49636
    for updates, episodes in ((8192, 0), (49636, 1)):
        changed = deepcopy(payload)
        changed['training_state']['metrics'].update(world_model_initialization_updates=updates,
            online_episodes_completed=episodes)
        torch.save(changed, path)
        parent()
        monkeypatch.setattr(trial, 'PRETRAINED_SHA256', trial.sha256(path))
        with pytest.raises(ValueError, match='49636'):
            trial.validate_pretrained(tmp_path)
    monkeypatch.setattr(trial, 'PRETRAINED_SHA256', 'different')
    with pytest.raises(ValueError, match='frozen complete'):
        trial.validate_pretrained(tmp_path)


def test_audit_encoding_preserves_unreachable_distance_without_nan_json():
    values = dict(cache=np.array([np.inf, -np.inf, 1.]), tensor=torch.tensor([2.]))
    encoded = json.dumps(trial.plain(values), allow_nan=False)
    assert '__nonfinite_float__' in encoded
    assert trial.fingerprint(values) == trial.fingerprint(json.loads(encoded))


def fake_session():
    clock = SimpleNamespace(time=30.)
    def save(path):
        with gzip.open(path, 'wb') as stream:
            stream.write(f'<snapshot time="{clock.time}"><rngState state="fixed"/></snapshot>'.encode())
    def load(path):
        clock.time = 30.
    sim = SimpleNamespace(saveState=save, loadState=load)
    fields = dict(_time_ms=30000, _vehicles={'a': seed('a')}, _active={'a'}, _pending={'a'},
        travel_times={}, reward_accounting=RewardAccounting(1, 30.), metrics={'wall_seconds': 7., 'events': 3},
        _incident_lanes={}, _incident_edges={}, _incident_active=False, _flow_cache=(30000, np.zeros((4, 8))),
        _actual_routes={}, _commanded_routes={}, _route_cursors={}, _invalid=False)
    return SimpleNamespace(**fields, backend='libsumo', traffic_model='meso', _closed=False,
        _connection=SimpleNamespace(simulation=sim)), clock


def test_restore_recovers_python_controller_and_all_external_rng(tmp_path, monkeypatch):
    session, clock = fake_session()
    net = network()
    controller = trial.CausalShortestController(net, NextHopCandidates(net, traffic_model='meso'), 40.)
    monkeypatch.setattr(trial, 'simulator_observation', lambda s: dict(time=clock.time,
        missing_registered=[], teleports=[[], []], count=len(s._active)))
    snapshot = trial.capture_state(session, controller, tmp_path/'initial.xml.gz', demand={'seed': 42})
    expected = trial.fingerprint(trial.causal_state(session, controller))
    random.random(); np.random.random(); torch.rand(3)
    session._active.clear()
    session._vehicles['a']['current_edge'] = 2
    session._time_ms = 60000
    controller.router.costs += 5.
    clock.time = 60.
    proof = trial.restore_state(session, controller, snapshot, audit_path=tmp_path/'restored.xml.gz')
    assert all(proof[k] for k in ('sumo_state_equal', 'causal_state_equal', 'observation_equal'))
    assert trial.fingerprint(trial.causal_state(session, controller)) == expected
    monkeypatch.setattr(trial, 'simulator_observation', lambda s: dict(time=clock.time,
        missing_registered=['future'], teleports=[[], []], count=len(s._active)))
    with pytest.raises(ValueError, match='replay gate failed'):
        trial.restore_state(session, controller, snapshot, audit_path=tmp_path/'bad.xml.gz')
    assert json.loads((tmp_path/'bad.xml.audit.json').read_text())['missing_registered'] == ['future']


def test_cohort_cost_excludes_future_background_and_uses_true_arrival_time():
    rows = [dict(vehicle_id='a', arrived=True, arrival_time=40.),
            dict(vehicle_id='b', arrived=False), dict(vehicle_id='future', arrived=False)]
    cost = trial.fixed_cohort_cost(rows, ['a', 'b'], 10., 130.)
    assert cost['vehicle_seconds'] == 150. and cost['mean_minutes'] == 1.25
    assert cost['cohort_size'] == 2 and cost['timeout_penalty'] == 0
    del rows[0]['arrival_time']
    with pytest.raises(ValueError, match='observed timestamp'):
        trial.fixed_cohort_cost(rows, ['a', 'b'], 10., 130.)


def test_intervention_is_legal_once_then_same_causal_controller():
    class Controller:
        candidates = SimpleNamespace(candidates=lambda *args: (1, 2))
        def __call__(self, rows, flow, now):
            return [NextHopDecision(row['vehicle_id'], row['current_edge'], 1, now) for row in rows]
    rows = [dict(vehicle_id='a', current_edge=0, destination_edge=3)]
    audit = []
    action = trial.intervention_controller(Controller(), target=rows[0], action=2, start_time=30., audit=audit)
    assert action(rows, None, 30.)[0].next_edge == 2
    assert action(rows, None, 31.)[0].next_edge == 1
    assert len(audit) == 1 and audit[0]['legal'] == [1, 2]
    invalid = trial.intervention_controller(Controller(), target=rows[0], action=3, start_time=30., audit=[])
    with pytest.raises(ValueError, match='legality'):
        invalid(rows, None, 30.)


class ConstantModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.dynamics = SimpleNamespace(graph_prior=SimpleNamespace(rank=1))
        self.conditions = []
    def encode(self, flow):
        return torch.zeros((*flow.shape[:-1], 64))
    def decode(self, latent):
        result = torch.zeros((*latent.shape[:-1], 4))
        result[..., 1] = 1.
        return result
    def transition(self, latent, action, *, action_events, noise, **kwargs):
        self.conditions.append((action.clone(), action_events, noise.clone()))
        return latent


@pytest.mark.parametrize('mode', ['sampled_prior_fm', 'zero_noise_prior_fm'])
def test_model_condition_events_are_own_causal_events_and_births_never_enter_cohort(mode):
    net = network((1000., 1000., 1000., 1000.))
    active = seed('active', current_edge=1, destination_edge=3, pending_decision=True,
        committed_next_edge=3, remaining_distance=1000.)
    snapshot = SimpleNamespace(session=dict(_time_ms=30000, _flow_cache=(30000, np.zeros((4, 8))),
        _vehicles={'active': active, 'future1': dict(departed=False, arrived=False),
                   'future2': dict(departed=False, arrived=False)}))
    model = ConstantModel()
    result = trial.model_branch(model, snapshot, SimpleNamespace(network=net), table(net),
        horizon=30, target=active, action=3, mode=mode, draw_seed=12, steps=2, executor_seed=8)
    action, events, noise = model.conditions[0]
    assert events.vehicle_ids == ('active',)
    assert action[0, 3] == pytest.approx(1/3)  # training's full-episode action denominator
    assert result['mean_minutes'] == .5  # fixed active-cohort cost denominator
    assert len(result['intervention']) == 1
    assert bool(noise.any()) == (mode == 'sampled_prior_fm')
    assert result['windows'][0]['own_action_events'][0]['vehicle_id'] == 'active'


def test_finite_action_ranking_and_regret_never_hide_missing_arms():
    assessment = trial.finite_action_assessment({1: 2., 2: 3., 3: 1.}, {1: 1., 2: 4., 3: 2.})
    assert assessment['selected'] == 1 and assessment['real_best'] == 3
    assert assessment['regret_minutes'] == 1.
    assert assessment['mean_signed_bias_minutes'] == pytest.approx(1/3)
    tied = trial.finite_action_assessment({1: 2., 2: 2.}, {1: 3., 2: 3.})
    assert tied['real_rank_groups'] == tied['predicted_rank_groups'] == [[1, 2]]
    with pytest.raises(ValueError, match='complete finite action set'):
        trial.finite_action_assessment({1: 2., 2: 3.}, {1: 1.})
    with pytest.raises(ValueError, match='finite'):
        trial.finite_action_assessment({1: 2.}, {1: float('nan')})


def test_native_sumo_failure_is_retained_and_cannot_be_claimed_success(tmp_path, monkeypatch):
    output = tmp_path/'failure'
    monkeypatch.setattr(trial.subprocess, 'run', lambda *args, **kwargs:
        SimpleNamespace(returncode=-11, stdout='load started', stderr='SIGSEGV at loadState'))
    assert trial.supervised_main(['--output', str(output)]) == 139
    failure = json.loads((output/'process_failure.json').read_text())
    assert failure['native_signal'] == 11 and failure['comparisons_blocked']
    assert 'loadState' in (output/'process_stderr.log').read_text()
    with pytest.raises(ValueError, match='overwrite'):
        trial.supervised_main(['--output', str(output)])
