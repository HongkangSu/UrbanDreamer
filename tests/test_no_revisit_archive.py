from dataclasses import replace

import pytest

from test_episode_archive import metadata, state, window, save
from urban_dreamer.learning.episode_archive import EpisodeArchive, read_episode_archive, to_replay_episode


def history_state(time, history, *, done=False):
    result = state(time, done=done)
    result['vehicles'][0].update(visited_edges=history, current_edge=history[-1])
    return result


def test_no_revisit_history_roundtrip_and_replay_seed(tmp_path):
    archive = EpisodeArchive(metadata(forbid_revisit=True))
    archive.start(history_state(0., (0,)))
    archive.append(replace(window(completed=True), next_state=history_state(30., (0, 1, 2), done=True)))
    path = tmp_path / 'episode.npz'
    save(archive, path, completed=True, travel_times={'v0': 28.})
    result = read_episode_archive(path)
    assert result['states'][-1]['vehicles'][0]['visited_edges'] == [0, 1, 2]
    replay = to_replay_episode(path, expected_session_protocol='event')
    assert replay['transitions'][0].vehicles[0]['visited_edges'] == [0]


@pytest.mark.parametrize('history', [None, [0, 1, 0], [0, True], [0, 3], [[0]]])
def test_no_revisit_archive_rejects_missing_invalid_or_repeated_history(history):
    archive = EpisodeArchive(metadata(forbid_revisit=True))
    initial = state(0.)
    if history is not None:
        initial['vehicles'][0]['visited_edges'] = history
    with pytest.raises(ValueError, match='visited_edges'):
        archive.start(initial)


def test_no_revisit_archive_rejects_history_reset_at_next_window():
    archive = EpisodeArchive(metadata(forbid_revisit=True))
    archive.start(history_state(0., [0, 1]))
    with pytest.raises(ValueError, match='previous window prefix'):
        archive.append(replace(window(), next_state=history_state(30., [1])))
