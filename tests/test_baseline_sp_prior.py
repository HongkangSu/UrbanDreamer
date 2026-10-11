"""Behavior protocol checks for annealed priors across baseline data and imagination."""
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_toy_baselines_model_based import (active_snapshot, make_args, module,
                                           synthetic_window)
from urban_dreamer.learning.baselines import build_method, common
from urban_dreamer.learning.baselines.mbpo import MbpoMethod
from urban_dreamer.learning.replay import MacroTransition
from urban_dreamer.models.policy import NextHopActor, set_shortest_path_prior


def prior_args(**overrides):
    return SimpleNamespace(**(module().PRIOR_DEFAULTS | {'sp_prior_weight': 1.} | overrides))


def test_prior_schedule_releases_and_restores_next_episode():
    tool = module()
    actor = NextHopActor()
    method = SimpleNamespace(actor=actor)
    args = prior_args()
    assert tool.apply_episode_prior(method, args, 200) == 1.
    assert tool.apply_episode_prior(method, args, 600) == pytest.approx(.5)
    expected = tool.apply_episode_prior(method, args, 601)
    restored = SimpleNamespace(actor=NextHopActor())
    tool.apply_episode_prior(restored, args, 600)
    assert tool.apply_episode_prior(restored, args, 601) == expected
    assert tool.apply_episode_prior(method, args, 1000) == 0.
    assert tool.apply_episode_prior(method, args, 5000) == 0.


@pytest.mark.parametrize('key,value', [('sp_prior_weight', .5), ('sp_prior_temperature', 60.),
    ('sp_prior_hold_episodes', 100), ('sp_prior_off_episode', 1200)])
def test_prior_profile_rejects_resume_changes(key, value):
    tool = module()
    args = prior_args()
    saved = dict(tool.prior_profile(args), reward_objective=common.reward_objective(args))
    saved[key] = value
    with pytest.raises(ValueError, match=key):
        tool.validate_prior_resume(saved, args)
    tool.validate_prior_resume({}, SimpleNamespace(**tool.PRIOR_DEFAULTS, reward_objective='legacy_exposure'))


@pytest.mark.parametrize('fail', [False, True])
def test_dual_evaluation_labels_seeds_and_restores_prior(tmp_path, fail):
    tool = module()
    actor = NextHopActor()
    set_shortest_path_prior(actor, .75, 15.)
    method = SimpleNamespace(actor=actor, select=lambda *args, **kwargs: None)
    seen = []
    def run(select, *, seed, label, **kwargs):
        seen.append((seed, label, actor.prior_weight))
        if fail and label == 'deterministic_no_prior':
            raise RuntimeError('evaluation failed')
        return [], [], dict(policy=label, selection_score=-actor.prior_weight)
    runner = SimpleNamespace(run_episode=run)
    kwargs = dict(episode=400, seeds=[10, 11], phase='evaluation',
                  policies=('deterministic', 'deterministic_no_prior'))
    if fail:
        with pytest.raises(RuntimeError, match='evaluation failed'):
            tool.evaluate(runner, method, tmp_path, **kwargs)
    else:
        records = tool.evaluate(runner, method, tmp_path, **kwargs)
        assert seen == [(10, 'deterministic', .75), (11, 'deterministic', .75),
                        (10, 'deterministic_no_prior', 0.), (11, 'deterministic_no_prior', 0.)]
        assert tool.selection_score(records, prior_enabled=True) == 0.
        assert tool.selection_score(records, prior_enabled=False) == -.75
    assert actor.prior_weight == .75
    assert actor.prior_temperature_seconds == 15.


def test_mbpo_cache_invalidates_only_when_weight_changes():
    method = MbpoMethod.__new__(MbpoMethod)
    method.actor = NextHopActor()
    method.model_rows = deque([{'sp_prior_weight': 1.}])
    method.model_rows_prior_weight = 1.
    set_shortest_path_prior(method.actor, 1., 30.)
    method._generate_model_rows([])
    assert len(method.model_rows) == 1
    set_shortest_path_prior(method.actor, .9, 30.)
    method._generate_model_rows([])
    assert len(method.model_rows) == 0
    assert method.model_rows_prior_weight == .9
    method.model_rows.append({'sp_prior_weight': .9})
    method._sync_model_rows_prior()
    assert len(method.model_rows) == 1
    method.model_rows.append({'sp_prior_weight': 1.})
    with pytest.raises(ValueError, match='synthetic rows'):
        method._policy_updates([], [])


@pytest.mark.parametrize('name', ['dreamer', 'mbpo'])
def test_real_and_imagined_decisions_share_frozen_prior(tmp_path, monkeypatch, name):
    import urban_dreamer.learning.baselines.dreamer as dreamer
    import urban_dreamer.learning.baselines.mbpo as mbpo
    from urban_dreamer.learning.observations import build_actor_observations
    runner = common.EpisodeRunner(make_args(tmp_path, method=name))
    method = build_method(name, runner.args, runner)
    module().apply_episode_prior(method, prior_args(), 600)
    vehicle = active_snapshot(runner)
    flow = runner.network.extract_flow({})
    timestamp = method.deadline - 30.
    seen = []
    sample = method.actor.forward_logits
    def capture(*args, **kwargs):
        seen.append(method.actor.prior_weight)
        return sample(*args, **kwargs)
    monkeypatch.setattr(method.actor, 'forward_logits', capture)
    obs = build_actor_observations(policy_edge_features=flow[:, [2, 1]], vehicle_states=[vehicle],
        candidates=runner.candidates, episode_progress=0., active_vehicle_fraction=1.)
    method.select([vehicle], obs, flow, timestamp)
    def advance(states, **kwargs):
        kwargs['decide'](states, kwargs['frozen_flow'], kwargs['start_time'])
        return SimpleNamespace(action=np.zeros(runner.network.edge_count),
                               departure=np.zeros(runner.network.edge_count))
    if name == 'dreamer':
        transition = MacroTransition(flow=flow, next_flow=flow, action=np.zeros(runner.network.edge_count),
            departure=np.zeros(runner.network.edge_count), context=np.array([0., 1.]),
            active_vehicle_fraction=1., vehicles=(vehicle,), time=timestamp,
            terminal=False, episode=0, step_index=0)
        method.seed_buffer.append_episode([transition], total_vehicle_count=runner.n, total_travel_minutes=1.)
        monkeypatch.setattr(dreamer, 'advance_imagined_window', advance)
        monkeypatch.setattr(method, '_decode_flow', lambda h, z: flow.copy())
        rows, _ = method._imagine_rollout(0)
    else:
        monkeypatch.setattr(mbpo, 'advance_imagined_window', advance)
        monkeypatch.setattr(method, '_predict_transition', lambda *args: (flow.copy(), -.25))
        rows = method._rollout_from_seed(synthetic_window(runner, np.random.default_rng(0),
            time=timestamp, terminal=False, vehicles=(vehicle,)), 0)
    assert len(seen) >= 2 and all(weight == pytest.approx(.5) for weight in seen)
    assert rows and all(row['sp_prior_weight'] == pytest.approx(.5) for row in rows)
    logits = method._compute_logits(rows)
    assert logits.requires_grad
    if name == 'mbpo':
        method._sync_model_rows_prior()
        method.model_rows.extend(rows)
        state = method.state_dict()
        method.model_rows.clear()
        method.model_rows_prior_weight = 0.
        method.load_state_dict(state)
        assert method.model_rows_prior_weight == pytest.approx(.5)
        assert len(method.model_rows) == len(rows)


def test_checkpoint_records_python_and_cuda_rng(tmp_path, monkeypatch):
    import random
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    sentinel = [torch.tensor([1, 2, 3], dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: sentinel)
    path = tmp_path / 'rng.pt'
    before = random.getstate()
    common.save_checkpoint(path, method_name='gnac', episode=600, best=1., method_state={},
        args=prior_args(), prepared=SimpleNamespace(input_hashes={}),
        config=SimpleNamespace(semantic_fingerprint='test'))
    state = torch.load(path, weights_only=False)
    assert state['python_rng'] == before
    assert torch.equal(state['cuda_rng'][0], sentinel[0])
    assert state['args']['sp_prior_off_episode'] == 1000
