"""The shared sampler preserves collection timing and policy observations."""
from types import SimpleNamespace

import numpy as np
import pytest

from urban_dreamer.learning import sumo_sampling as sampling
from urban_dreamer.traffic.environment import NextHopDecision, WindowResult


class Session:
    def __init__(self, *, empty=False, fail_reset=False):
        self.done = empty
        self.fail_reset = fail_reset
        self.closed = False
        self.calls = 0
        self.active_vehicle_count = 2
        self.travel_times = {}

    def reset(self):
        if self.fail_reset:
            raise RuntimeError('reset failed')
        self.state = dict(time=0., flow=np.zeros((3, 4), dtype=np.float32),
                          active_vehicle_count=1, vehicles=[])
        return self.state

    def advance_window(self, decide):
        self.calls += 1
        old = self.state
        pending = [dict(vehicle_id='b'), dict(vehicle_id='a')]
        decisions = tuple(decide(pending, old['flow'], old['time'] + 1))
        decisions += tuple(decide(pending, old['flow'], old['time'] + 2))
        self.state = dict(old, time=old['time'] + 30)
        counts = np.zeros(3, dtype=np.int64)
        return WindowResult(decisions, counts, counts.copy(), 17., self.state,
                            old['time'], self.state['time'])

    def close(self):
        self.closed = True


@pytest.fixture
def observations(monkeypatch):
    calls = []
    real_prepare = sampling.prepare_actor_window
    prepared = []
    def prepare(*args):
        prepared.append(args)
        return real_prepare(*args)
    def build(**kwargs):
        calls.append(kwargs)
        return kwargs
    monkeypatch.setattr(sampling, 'prepare_actor_window', prepare)
    monkeypatch.setattr(sampling, 'build_actor_observations', build)
    return calls, prepared


def sampler(session, **kwargs):
    kwargs.setdefault('reward_objective', 'legacy_exposure')
    return sampling.SumoEpisodeSampler(session=session,
        candidates=SimpleNamespace(network=SimpleNamespace(edge_count=3)),
        deadline_seconds=60., total_vehicle_count=4, **kwargs)


def decide(pending, obs, frozen, now, index, context):
    return [NextHopDecision(v['vehicle_id'], 0, 1, now) for v in pending]


def test_window_cache_context_and_event_order(observations):
    session = Session()
    with sampler(session, context_mode='window') as collector:
        assert collector.state['time'] == 0
        rows = list(collector.windows(decide))
        assert collector.state['time'] == 60
    calls, prepared = observations
    assert len(rows) == 2 and session.calls == 2 and session.closed
    assert len(prepared) == 2 and len(calls) == 4
    assert calls[0]['policy_edge_features'] is calls[1]['policy_edge_features']
    assert calls[1]['policy_edge_features'] is not calls[2]['policy_edge_features']
    assert [c['episode_progress'] for c in calls] == [0, 0, .5, .5]
    assert [c['active_vehicle_fraction'] for c in calls] == [.25] * 4
    assert [d.vehicle_id for d in rows[0].result.decisions] == ['b', 'a', 'b', 'a']
    assert rows[0].result.active_vehicle_seconds == 17
    assert rows[0].start_state['time'] == 0 and rows[1].index == 1
    with pytest.raises(Exception):
        rows[0].index = 10


def test_observation_context_sorted_pending(observations):
    with sampler(Session(), context_mode='observation', sort_pending=True, max_windows=1) as collector:
        rows = list(collector.windows(decide))
    calls, _ = observations
    assert calls[0]['episode_progress'] == pytest.approx(1 / 60)
    assert calls[1]['episode_progress'] == pytest.approx(2 / 60)
    assert calls[0]['active_vehicle_fraction'] == .5
    assert [d.vehicle_id for d in rows[0].result.decisions] == ['a', 'b', 'a', 'b']


@pytest.mark.parametrize('pad,expected', [(False, 0), (True, 1)])
def test_empty_terminal_padding(pad, expected, observations):
    session = Session(empty=True)
    with sampler(session, pad_empty=pad) as collector:
        rows = list(collector.windows(decide))
    assert len(rows) == expected and session.calls == 0
    if rows:
        assert rows[0].result.end_time == 30
        assert rows[0].result.next_state['done']
        assert rows[0].result.active_vehicle_seconds == 0
        assert not rows[0].result.action_counts.any()
    assert not observations[0]


def test_close_on_callback_and_reset_failure(observations):
    for reset_failure in (False, True):
        session = Session(fail_reset=reset_failure)
        with pytest.raises(RuntimeError):
            with sampler(session) as collector:
                def fail(*args):
                    raise RuntimeError('policy failed')
                list(collector.windows(fail))
        assert session.closed


def test_timeout_closes_without_advance(observations, monkeypatch):
    ticks = iter([10., 12.])
    monkeypatch.setattr(sampling.time, 'monotonic', lambda: next(ticks))
    session = Session()
    with pytest.raises(TimeoutError, match='online episode exceeded'):
        with sampler(session, wall_timeout_seconds=1., timeout_label='online') as collector:
            list(collector.windows(decide))
    assert session.closed and session.calls == 0


def test_factory_legacy_default_and_event_backend():
    assert sampling.resolve_session_factory('microstep', 1.) is None
    factory = sampling.resolve_session_factory('event', 1., 'libsumo')
    assert factory.keywords == dict(step_seconds=1., backend='libsumo')
    with pytest.raises(ValueError):
        sampling.resolve_session_factory('invalid', 1.)


def test_initial_done_can_use_session_terminal_transition(observations):
    session = Session(empty=True)
    with sampler(session, pad_empty=False, advance_empty=True) as collector:
        rows = list(collector.windows(decide))
    assert len(rows) == 1 and session.calls == 1
    assert rows[0].result.active_vehicle_seconds == 17


def test_before_window_captures_buffers_before_session_mutates(observations):
    session = Session()
    original = session.advance_window
    def mutate(decide):
        session.state['flow'][:] += 1
        return original(decide)
    session.advance_window = mutate
    captured = []
    def before(index, state, context):
        captured.append((index, state['flow'].copy(), context.copy()))
    with sampler(session) as collector:
        rows = list(collector.windows(decide, before_window=before))
    assert len(rows) == 2
    assert np.all(captured[0][1] == 0) and np.all(captured[1][1] == 1)
    assert captured[1][0] == 1


def test_context_manager_enter_exit_once_and_exception_type(observations):
    session = Session()
    class Manager:
        def __init__(self):
            self.enters = 0
            self.exits = []
        def __enter__(self):
            self.enters += 1
            return session
        def __exit__(self, *error):
            self.exits.append(error)
            session.close()
    manager = Manager()
    with pytest.raises(RuntimeError, match='policy'):
        with sampler(manager) as collector:
            def fail(*args):
                raise RuntimeError('policy')
            list(collector.windows(fail))
    assert manager.enters == 1 and len(manager.exits) == 1
    assert manager.exits[0][0] is RuntimeError and session.closed


def test_terminal_session_stops_before_deadline(observations):
    session = Session()
    original = session.advance_window
    def terminal(decide):
        result = original(decide)
        session.done = True
        return result
    session.advance_window = terminal
    with sampler(session) as collector:
        assert len(list(collector.windows(decide))) == 1
        assert collector.state['time'] == 30
        with pytest.raises(RuntimeError, match='only be iterated once'):
            list(collector.windows(decide))


def test_default_paper_sampler_uses_live_started_history_not_future_population(observations):
    from urban_dreamer.learning.global_reward import RewardAccounting

    class AccountingSession(Session):
        def __init__(self):
            super().__init__()
            self.active_vehicle_count = 1
            self.reward_accounting = RewardAccounting(1, 0.)

        def advance_window(self, decide):
            old = self.state
            pending = [dict(vehicle_id='a')]
            self.reward_accounting.update(1., 1)
            first = tuple(decide(pending, old['flow'], 1.))
            self.active_vehicle_count = 2
            self.reward_accounting.update(1., 2)
            second = tuple(decide(pending, old['flow'], 2.))
            self.reward_accounting.update(15., 2)
            self.state = dict(old, time=30., active_vehicle_count=2)
            counts = np.zeros(3, dtype=np.int64)
            return WindowResult(first + second, counts, counts.copy(), 17., self.state, 0., 30.)

    captures = []
    for future_population in (4, 400):
        session = AccountingSession()
        with sampling.SumoEpisodeSampler(session=session,
                candidates=SimpleNamespace(network=SimpleNamespace(edge_count=3)),
                deadline_seconds=60., total_vehicle_count=future_population,
                max_windows=1) as collector:
            def capture(pending, obs, frozen, now, index, context):
                captures.append(context.copy())
                return decide(pending, obs, frozen, now, index, context)
            list(collector.windows(capture))
            assert collector.accounting.started_count == 2
            assert collector.accounting.cumulative_vehicle_seconds == 17.
    np.testing.assert_array_equal(captures[:2], captures[2:])
    assert captures[0][1] == captures[1][1] == 1.
    assert captures[0][3] == pytest.approx(np.log1p(1))
    assert captures[1][3] == pytest.approx(np.log1p(2))
