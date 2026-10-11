"""Physical candidate network edits preserve their explicitly scoped semantics."""

import xml.etree.ElementTree as ET

import pytest

from tools.patch_candidate_uturns import randomize_short_lanes, reciprocal_turn_pairs


def test_bidi_bound_short_edges_share_random_length_and_repair_existing_mismatch():
    from tools.patch_candidate_uturns import harmonize_bidi_lengths
    root = ET.fromstring('''<net>
      <edge id="east" bidi="west"><lane id="east_0" index="0" length="10"/></edge>
      <edge id="west" bidi="east"><lane id="west_0" index="0" length="10"/></edge>
    </net>''')
    randomize_short_lanes(root, 'city')
    a, b = root.findall('edge/lane')
    assert 10 < float(a.get('length')) < 15
    assert a.get('length') == b.get('length')
    b.set('length', '11.123')
    changes = harmonize_bidi_lengths(root)
    assert len(changes) == 1
    assert a.get('length') == b.get('length')
    assert harmonize_bidi_lengths(root) == []


def network(edges, connections=()):
    root = ET.Element('net')
    for edge_id, lanes, attrs in edges:
        edge = ET.SubElement(root, 'edge', id=edge_id, **attrs)
        for index, lane_attrs in enumerate(lanes):
            ET.SubElement(edge, 'lane', id=f'{edge_id}_{index}', index=str(index),
                          **{'length': '10', **lane_attrs})
    for source, target, attrs in connections:
        ET.SubElement(root, 'connection', **{
            'from': source, 'to': target, 'fromLane': '0', 'toLane': '0',
            'dir': 't', **attrs,
        })
    return root


def test_reciprocal_turns_require_two_actual_uturn_connections():
    root = network(
        [(name, [{}], {}) for name in ('a', '-a', 'b', '-b', 'c', '-c')],
        [('a', '-a', {}), ('-a', 'a', {}),
         ('b', '-b', {}), ('-b', 'b', {'dir': 's'}),
         ('c', '-c', {})],
    )
    assert reciprocal_turn_pairs(root) == {('-a', 'a')}


def test_reciprocal_turns_use_connection_lanes_and_deduplicate_multilane_pairs():
    root = network(
        [('a', [{}, {}], {}), ('-a', [{}, {}], {}),
         ('b', [{'allow': 'bus'}, {}], {}), ('-b', [{}, {}], {})],
        [('a', '-a', {}), ('a', '-a', {'fromLane': '1', 'toLane': '1'}),
         ('-a', 'a', {}), ('-a', 'a', {'fromLane': '1', 'toLane': '1'}),
         ('b', '-b', {}), ('-b', 'b', {'toLane': '1'})],
    )
    # A passenger-capable second lane does not legalize a connection from lane 0.
    assert reciprocal_turn_pairs(root) == {('-a', 'a')}


@pytest.mark.parametrize('forbidden', [{'allow': 'bus'}, {'disallow': 'passenger'},
                                      {'disallow': 'all'}])
def test_nonpassenger_source_or_destination_prevents_reciprocal_pair(forbidden):
    root = network(
        [('a', [forbidden], {}), ('-a', [{}], {})],
        [('a', '-a', {}), ('-a', 'a', {})],
    )
    assert reciprocal_turn_pairs(root) == set()


def test_explicit_passenger_permission_is_accepted():
    root = network(
        [('a', [{'allow': 'passenger bus'}], {}),
         ('-a', [{'disallow': 'bus bicycle'}], {})],
        [('a', '-a', {}), ('-a', 'a', {})],
    )
    assert reciprocal_turn_pairs(root) == {('-a', 'a')}


def short_network():
    return network([
        ('a', [{'length': '5'}, {'length': '10'}, {'length': '10.01'}], {}),
        ('b', [{'length': '9.9'}], {}),
        ('long', [{'length': '30'}], {}),
        (':internal', [{'length': '2'}], {}),
        ('internal_function', [{'length': '3'}], {'function': 'internal'}),
    ])


def lane_lengths(root):
    return {lane.get('id'): float(lane.get('length')) for lane in root.findall('edge/lane')}


def test_short_lane_randomization_includes_ten_and_shares_per_edge_value():
    root = short_network()
    before = lane_lengths(root)
    changes = randomize_short_lanes(root, 'coulommiers', seed=42)
    after = lane_lengths(root)
    assert {row['lane'] for row in changes} == {'a_0', 'a_1', 'b_0'}
    assert after['a_0'] == after['a_1']
    for row in changes:
        assert row['edge'] in {'a', 'b'}
        assert row['old_length_m'] == before[row['lane']]
        assert row['new_length_m'] == after[row['lane']]
        assert 10 < row['new_length_m'] <= 15
    for lane in ('a_2', 'long_0', ':internal_0', 'internal_function_0'):
        assert before[lane] == after[lane]


def test_short_lane_randomization_reproducible_and_seed_sensitive():
    first, second, different = short_network(), short_network(), short_network()
    changes = randomize_short_lanes(first, 'coulommiers', seed=42)
    assert randomize_short_lanes(second, 'coulommiers', seed=42) == changes
    randomize_short_lanes(different, 'coulommiers', seed=43)
    assert lane_lengths(first) == lane_lengths(second)
    assert lane_lengths(first) != lane_lengths(different)
    assert randomize_short_lanes(first, 'coulommiers', seed=42) == []
def test_drop_unreachable_od_preserves_surviving_rows_and_weights(tmp_path, monkeypatch):
    import csv
    from tools import patch_candidate_uturns as patch
    fields = ['id', 'origin', 'destination', 'start_time']
    rows = [dict(zip(fields, values)) for values in (
        ('0', '1', '2', '7.25'), ('1', '3', '4', '10'),
        ('2', '1', '2', '11'), ('3', '5', '6', '16'))]
    with (tmp_path/'agents.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    monkeypatch.setattr(patch, 'check_od', lambda directory, city: dict(
        unreachable_od_pairs=[dict(origin_index=1, destination_index=2, expected_vehicles=2)]))
    result = patch.drop_unreachable_od(tmp_path, 'test')
    with (tmp_path/'agents.csv').open(newline='') as stream:
        assert list(csv.DictReader(stream)) == [rows[1], rows[3]]
    assert result['remaining_od_pairs'] == 2
    assert result['remaining_expected_vehicles'] == 2
    assert result['removed_expected_vehicles'] == 2
