"""Route policy timing and observed turn conditioning share current SUMO physics."""
import numpy as np
import pytest

from test_event_session_commitments import session_at
from test_meso_event_session import meso_setup
from test_episode_archive import metadata, state, save
from urban_dreamer.learning.episode_archive import EpisodeArchive, read_episode_archive, to_replay_episode
from urban_dreamer.traffic.environment import RouteDecision, NextHopDecision, WindowResult
from urban_dreamer.traffic.event_session import SumoEventSession
from urban_dreamer.traffic.route_candidates import RouteCandidates


@pytest.mark.parametrize("meso", [False, True])
def test_observed_fork_keeps_selected_suffix_without_requesting_policy(meso):
    session = session_at('b', ('a', 'b', 'c', 'b', 'd'), 1)
    session.routing_mode = 'current_to_dest'
    session.traffic_model = 'meso' if meso else 'micro'
    session._actual_routes = {'v': ('a', 'b', 'c', 'b', 'd')}
    session._observe(np.zeros(4))
    row = session._vehicles['v']
    assert row['committed_route'] == (1, 2, 1, 3)
    assert row['committed_next_edge'] == 2
    assert not row['pending_decision'] and not session._pending
    turns = session._conditioning_turns()
    assert turns == [NextHopDecision('v', 1, 2, 30.)]
    assert not session._conditioning_turns(), 'one observation produces one conditioning turn'


def test_skipped_fork_does_not_create_a_conditioning_event():
    session = session_at('c', ('a', 'b', 'c', 'b', 'd'), 2)
    session.routing_mode = 'current_to_dest'
    session._actual_routes = {'v': ('a', 'b', 'c', 'b', 'd')}
    session._observe(np.zeros(4))
    assert session._vehicles['v']['committed_route'] == (2, 1, 3)
    assert not session._conditioning_turns()


def test_route_archives_keep_turn_conditions_and_policy_actions_separately(tmp_path):
    archive = EpisodeArchive(metadata(config={'simulation': {'routing_mode': 'current_to_dest'}}))
    archive.start(state(0.))
    policy = RouteDecision('v0', 0, 1, 0., (0, 1, 2))
    turn = NextHopDecision('v0', 1, 2, 17.)
    result = WindowResult((policy,), np.array([0., 0., 1.]), np.zeros(3), 30.,
                          state(30.), 0., 30., conditioning_decisions=(turn,))
    archive.append(result)
    path = tmp_path / 'route.npz'
    save(archive, path)
    data = read_episode_archive(path)
    assert data['metadata']['routing_mode'] == 'current_to_dest'
    assert data['metadata']['action_semantics'] == 'branch-only'
    assert data['metadata']['policy_action_semantics'] == 'periodic-current-to-dest'
    assert [(row['current_edge'], row['next_edge']) for row in data['decisions']] == [(1, 2)]
    assert data['policy_decisions'][0]['route'] == [0, 1, 2]
    assert data['policy_decisions'][0]['decision_time'] == 0.
    replay = to_replay_episode(path, expected_session_protocol='event')
    assert replay['transitions'][0].action_events[0].decision_time == 17.
    np.testing.assert_array_equal(replay['transitions'][0].action, [0., 0., 1.])


def test_old_archives_default_policy_audit_to_existing_decisions(tmp_path):
    archive = EpisodeArchive(metadata())
    archive.start(state(0.))
    archive.append(WindowResult((NextHopDecision('v0', 0, 1, 1.),), np.array([0., 1., 0.]),
                               np.zeros(3), 30., state(30.), 0., 30.))
    path = tmp_path / 'old.npz'
    save(archive, path)
    with np.load(path, allow_pickle=False) as stored:
        assert 'policy_decisions_json' not in stored.files
    data = read_episode_archive(path)
    assert 'routing_mode' not in data['metadata'] and 'policy_action_semantics' not in data['metadata']
    assert data['policy_decisions'] == data['decisions']


def test_real_meso_route_policy_runs_only_at_boundaries_and_conditions_observed_forks(tmp_path):
    config, prepared, topology = meso_setup(8)
    config.simulation.routing_mode = 'current_to_dest'
    calls, policy_actions, turns = [], [], []
    with SumoEventSession(config, prepared, topology, backend='libsumo') as session:
        archive = EpisodeArchive(metadata(population=8, physics_identity=session.physics,
            config={'simulation': {'traffic_model': 'meso', 'routing_mode': 'current_to_dest'}}))
        archive.start(session.reset())
        api = session._connection.vehicle

        def policy(rows, flow, now):
            calls.append(now)
            assert now % 30 == 0
            result = []
            for row in rows:
                assert row['departed'] and not row['arrived'] and not row['in_junction']
                paths = session.candidates.generate_paths(row['current_edge'], row['destination_edge'])
                route = paths[1]
                result.append(RouteDecision(row['vehicle_id'], row['current_edge'], route[1], now, route))
            return result

        for index in range(30):
            result = session.advance_window(policy)
            archive.append(result)
            if index == 0:
                assert not result.decisions and not calls, 'departure waits for next policy boundary'
            policy_actions.extend(result.decisions)
            turns.extend(result.conditioning_decisions)
            assert result.action_counts.sum() == len(result.conditioning_decisions)
            assert all(result.start_time <= event.decision_time < result.end_time
                       for event in result.conditioning_decisions)
            for vid in session._active:
                row = session._vehicles[vid]
                route, cursor = api.getRoute(vid), api.getRouteIndex(vid)
                expected = tuple(session.network.edge_to_index[edge] for edge in route[cursor:])
                assert row['committed_route'] == expected
                assert not row['pending_decision']
            if session.done:
                break
        assert session.done
        assert calls and len(calls) == len(set(calls))
        assert turns and any(turn.decision_time % 30 != 0 for turn in turns)
        assert policy_actions and all(action.route[0] == action.current_edge for action in policy_actions)
        path = tmp_path / 'real_route.npz'
        archive.save(path, metrics=session.metrics, travel_times=session.travel_times,
                     completed=True, deadline_seconds=900.)
    data = read_episode_archive(path)
    assert len(data['decisions']) == len(turns)
    assert len(data['policy_decisions']) == len(policy_actions)
    assert data['action_counts'].sum() == len(turns)


@pytest.mark.parametrize('retained_prefix', [False, True])
def test_queued_boundary_turn_uses_route_replacement_and_emits_once(retained_prefix):
    session = session_at('b', ('a', 'b', 'd'), 1, current=1, pending=True)
    session.routing_mode = 'current_to_dest'
    session.candidates = RouteCandidates(session.network)
    session._vehicles['v']['pending_turn'] = True
    route = session.candidates.generate_paths(1, 3)[0]
    actual = list(('a', 'b', 'd'))
    session._actual_routes = {'v': tuple(actual)}
    session._connection.vehicle.setRoute = lambda vid, value: actual.__setitem__(
        slice(None), ['a', *value] if retained_prefix else value)
    session._connection.vehicle.getRoute = lambda vid: actual
    session._connection.vehicle.getRouteIndex = lambda vid: int(retained_prefix)
    session._apply_decisions(lambda rows, flow, now: [RouteDecision('v', 1, route[1], now, route)],
                             np.zeros((4, 8)))
    assert session._vehicles['v']['committed_route'] == route
    assert session._conditioning_turns() == [NextHopDecision('v', 1, route[1], 30.)]
    assert not session._conditioning_turns()


def test_real_meso_accepts_different_suffix_with_the_same_first_hop():
    config, prepared, topology = meso_setup(1)
    config.simulation.routing_mode = 'current_to_dest'
    config.simulation.vehicle_max_speed_mps = 1.
    selected = []
    with SumoEventSession(config, prepared, topology, backend='libsumo') as session:
        session.reset()

        def choose_alternative(rows, flow, now):
            assert now == 30.
            row = rows[0]
            paths = session.candidates.generate_paths(row['current_edge'], row['destination_edge'])
            assert paths[0] != paths[1] and paths[0][1] == paths[1][1]
            selected.append(paths[1])
            return [RouteDecision(row['vehicle_id'], row['current_edge'], paths[1][1], now, paths[1])]

        first = session.advance_window(choose_alternative)
        assert not selected and not first.decisions
        second = session.advance_window(choose_alternative)
        assert len(second.decisions) == 1
        row = second.next_state['vehicles'][0]
        assert row['committed_route'] == selected[0]
        api = session._connection.vehicle
        actual = api.getRoute(row['vehicle_id'])
        cursor = api.getRouteIndex(row['vehicle_id'])
        assert tuple(session.network.edge_to_index[edge] for edge in actual[cursor:]) == selected[0]
