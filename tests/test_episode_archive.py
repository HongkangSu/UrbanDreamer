from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from urban_dreamer.learning.episode_archive import EpisodeArchive, read_episode_archive, to_replay_episode
from urban_dreamer.traffic.environment import NextHopDecision, WindowResult


def metadata(population=1, **extra):
    return dict(city="toy", method="random", seed=7, episode=0, phase="online",
                input_hashes={"network": "test"}, total_vehicle_count=population,
                session_protocol="event", demand_realization=[], **extra)


def state(timestamp, *, population=1, done=False):
    return {"time": timestamp, "flow": np.full((3, 8), timestamp, dtype=np.float32),
            "done": done, "vehicles": [{"vehicle_id": "v0", "departed": True,
                "arrived": done, "current_edge": 0, "destination_edge": 2,
                "remaining_distance": 12., "departure_time": 0.,
                "committed_next_edge": 1, "in_junction": False,
                "junction_remaining_distance": 0., "junction_speed": 0.}][:population]}


def window(*, completed=False):
    return WindowResult((NextHopDecision("v0", 0, 1, 2.), NextHopDecision("v0", 1, 2, 20.)),
                        np.array([0., 1., 1.]), np.zeros(3), 30., state(30., done=completed), 0., 30.)


def save(archive, path, *, completed=False, travel_times=None):
    archive.save(path, metrics={"metric": 2.}, travel_times=travel_times or {},
                 completed=completed, deadline_seconds=30.)


def test_multiple_events_and_snapshots_roundtrip(tmp_path):
    archive = EpisodeArchive(metadata())
    initial = state(0.)
    archive.start(initial)
    initial["vehicles"][0]["remaining_distance"] = 999.
    archive.append(window(completed=True))
    path = tmp_path / "episode.npz"
    save(archive, path, completed=True, travel_times={"v0": 28.})
    with np.load(path, allow_pickle=False) as stored:
        assert all(stored[key].dtype != object for key in stored.files)
    data = read_episode_archive(path, expected_session_protocol="event")
    assert data["flow"].shape == (2, 3, 8)
    assert data["flow"].dtype == np.float32
    assert [row["decision_time"] for row in data["decisions"]] == [2., 20.]
    assert [row["current_edge"] for row in data["decisions"]] == [0, 1]
    assert data["states"][0]["vehicles"][0]["remaining_distance"] == 12.
    assert data["states"][-1]["vehicles"][0]["arrived"]
    assert data["terminated"].tolist() == [True]
    assert data["truncated"].tolist() == [False]
    replay = to_replay_episode(path, expected_session_protocol="event")
    assert replay["transitions"][0].terminal
    assert replay["total_travel_minutes"] == 28. / 60.


def test_censored_episode_remains_nonterminal_with_observed_exposure(tmp_path):
    archive = EpisodeArchive(metadata())
    archive.start(state(0.))
    archive.append(window())
    path = tmp_path / "episode.npz"
    save(archive, path)
    data = read_episode_archive(path)
    assert data["terminated"].tolist() == [False]
    assert data["truncated"].tolist() == [True]
    replay = to_replay_episode(path, expected_session_protocol="event")
    assert not replay["transitions"][0].terminal
    assert replay["total_travel_minutes"] == .5
    assert replay["total_vehicle_count"] == 1


def test_zero_population_retains_one_physical_window(tmp_path):
    archive = EpisodeArchive(metadata(population=0))
    archive.start(state(0., population=0, done=True))
    archive.append(WindowResult((), np.zeros(3), np.zeros(3), 0.,
                                state(30., population=0, done=True), 0., 30.))
    path = tmp_path / "zero.npz"
    save(archive, path, completed=True)
    replay = to_replay_episode(path, expected_session_protocol="event")
    assert len(replay["transitions"]) == 1
    assert replay["transitions"][0].terminal
    assert replay["total_vehicle_count"] == 0
    assert replay["total_travel_minutes"] == 0
    assert replay["transitions"][0].action.sum() == 0


def test_actual_population_normalizes_actions_not_active_population(tmp_path):
    from copy import deepcopy
    initial, final = state(0.), state(30.)
    for boundary in (initial, final):
        future = deepcopy(boundary["vehicles"][0])
        future.update(vehicle_id="future", departed=False, pending_decision=False, departure_time=60.)
        boundary["vehicles"].append(future)
    archive = EpisodeArchive(metadata(population=2))
    archive.start(initial)
    archive.append(replace(window(), next_state=final))
    path = tmp_path / "population.npz"
    save(archive, path)
    data = read_episode_archive(path)
    np.testing.assert_array_equal(data["action_counts"], [[0, 1, 1]])
    np.testing.assert_array_equal(data["action"], [[0, .5, .5]])
    replay = to_replay_episode(path, expected_session_protocol="event")
    assert replay["total_vehicle_count"] == 2
    assert replay["total_travel_minutes"] == .25
    assert len(replay["transitions"][0].vehicles) == 1
    assert replay["transitions"][0].context[1] == .5


@pytest.mark.parametrize("bad", [
    lambda result: replace(result, next_state={**result.next_state, "flow": np.zeros((4, 8))}),
    lambda result: replace(result, start_time=1.),
    lambda result: replace(result, action_counts=np.zeros(3)),
    lambda result: replace(result, active_vehicle_seconds=float("nan")),
    lambda result: replace(result, decisions=(NextHopDecision("v0", 0, 1, 30.),)),
])
def test_invalid_window_rejected_without_partial_append(bad, tmp_path):
    archive = EpisodeArchive(metadata())
    archive.start(state(0.))
    with pytest.raises(ValueError):
        archive.append(bad(window()))
    archive.append(window())
    path = tmp_path / "valid.npz"
    save(archive, path)
    assert len(read_episode_archive(path)["action"]) == 1


def test_protocol_explicit_and_mismatch_rejected(tmp_path):
    archive = EpisodeArchive(metadata())
    archive.start(state(0.))
    archive.append(window())
    path = tmp_path / "episode.npz"
    save(archive, path)
    with pytest.raises(ValueError, match="protocol mismatch"):
        to_replay_episode(path, expected_session_protocol="microstep")
    with pytest.raises(ValueError, match="explicit"):
        to_replay_episode(path, expected_session_protocol=None)
    with pytest.raises(TypeError):
        to_replay_episode(path)


def test_atomic_retry_and_provenance_protection(tmp_path, monkeypatch):
    archive = EpisodeArchive(metadata())
    archive.start(state(0.))
    archive.append(window())
    path = tmp_path / "episode.npz"
    save(archive, path)
    before = path.read_bytes()
    save(archive, path)
    assert read_episode_archive(path)["metadata"]["seed"] == 7
    archive.metadata["seed"] = 8
    with pytest.raises(ValueError, match="provenance"):
        save(archive, path)
    assert path.read_bytes() == before
    archive.metadata["seed"] = 7
    def fail(*args, **kwargs):
        raise OSError("disk failure")
    monkeypatch.setattr(np, "savez_compressed", fail)
    with pytest.raises(OSError):
        save(archive, path)
    assert path.read_bytes() == before
    missing = tmp_path / "missing.npz"
    with pytest.raises(OSError):
        save(archive, missing)
    assert not missing.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["episode.npz"]
