"""Short-loop restrictions preserve the actual passenger demand graph."""
import xml.etree.ElementTree as ET
import json
import shutil
import sys

import networkx as nx
import pytest

from tools.patch_short_road_cycles import (apply_extra_removals, passenger_graph, plan_removals,
                                          signal_signature,
                                          unreachable_od, validate_signal_equivalence)


def network(lengths, connections, permissions=None):
    root = ET.Element('net')
    for name, length in lengths.items():
        edge = ET.SubElement(root, 'edge', id=name)
        ET.SubElement(edge, 'lane', id=f'{name}_0', index='0', length=str(length),
                      **(permissions or {}).get(name, {}))
    for source, target, direction in connections:
        ET.SubElement(root, 'connection', **{'from': source, 'to': target,
            'fromLane': '0', 'toLane': '0', 'dir': direction})
    return root


def apply_plan(root, result):
    graph, _, _ = passenger_graph(root)
    for row in result['removed_pairs']:
        graph.remove_edge(row['source'], row['target'])
    return graph


def test_three_road_cycle_with_alternate_route_preserves_demand():
    root = network(dict(a=12, b=12, c=12, x=100),
        [('a', 'b', 'l'), ('b', 'c', 's'), ('c', 'a', 's'),
         ('a', 'x', 's'), ('x', 'b', 's')])
    od = [('a', 'c'), ('b', 'a'), ('c', 'b')]
    result = plan_removals(root, od)
    assert len(result['removed_pairs']) == 1
    assert result['removed_pairs'][0]['source'] == 'a'
    assert result['removed_pairs'][0]['target'] == 'b'
    assert result['removed_pairs'][0]['globally_redundant']
    graph = apply_plan(root, result)
    assert not unreachable_od(graph, od)
    assert nx.has_path(graph, 'a', 'b')
    assert result['stages'][0]['cycles_remaining'] == 0


def test_essential_cycle_is_reported_and_preserved():
    root = network(dict(a=12, b=12, c=12),
        [('a', 'b', 'l'), ('b', 'c', 'r'), ('c', 'a', 's')])
    od = [('a', 'b'), ('b', 'c'), ('c', 'a')]
    result = plan_removals(root, od)
    assert result['removed_pairs'] == []
    assert result['stages'][0]['cycles_remaining'] == 1
    assert result['stages'][0]['remaining_cycles'] == [['a', 'b', 'c']]
    assert len(result['rejected_pairs']) == 3


def test_od_safe_deletion_does_not_require_global_redundancy():
    root = network(dict(a=12, b=12, c=12, destination=100),
        [('a', 'b', 't'), ('b', 'c', 's'), ('c', 'a', 's'),
         ('a', 'destination', 's'), ('b', 'destination', 's'), ('c', 'destination', 's')])
    od = [(name, 'destination') for name in ('a', 'b', 'c')]
    result = plan_removals(root, od)
    assert len(result['removed_pairs']) == 1
    assert not result['removed_pairs'][0]['globally_redundant']
    assert not unreachable_od(apply_plan(root, result), od)


@pytest.mark.parametrize('permission', [{'allow': 'bus'}, {'disallow': 'passenger'}])
def test_nonpassenger_lane_connections_do_not_form_cycles(permission):
    root = network(dict(a=12, b=12, c=12),
        [('a', 'b', 'l'), ('b', 'c', 's'), ('c', 'a', 's')], {'b': permission})
    # A different allowed lane cannot legalize the connection from lane 0.
    edge = next(edge for edge in root.findall('edge') if edge.get('id') == 'b')
    ET.SubElement(edge, 'lane', id='b_1', index='1', length='12', allow='passenger')
    result = plan_removals(root, [('c', 'a')])
    assert result['removed_pairs'] == []
    assert result['stages'][0]['cycles_before'] == 0


def test_connection_order_and_edge_order_do_not_change_plan():
    lengths = dict(a=12, b=12, c=12, d=20, x=100)
    connections = [('a', 'b', 'l'), ('b', 'c', 's'), ('c', 'a', 's'),
                   ('b', 'd', 'r'), ('d', 'a', 's'), ('a', 'x', 's'), ('x', 'b', 's')]
    first = network(lengths, connections)
    second = network(dict(reversed(list(lengths.items()))), list(reversed(connections)))
    od = [('a', 'c'), ('b', 'a'), ('c', 'b')]
    assert plan_removals(first, od) == plan_removals(second, list(reversed(od)))


def test_cycles_outside_length_and_local_size_bounds_are_untouched():
    lengths = {name: 20 for name in 'abcdef'} | dict(x=100, y=12, z=12)
    connections = [(a, b, 'l') for a, b in zip('abcdef', 'bcdefa')]
    connections += [('x', 'y', 't'), ('y', 'z', 'l'), ('z', 'x', 'r')]
    root = network(lengths, connections)
    result = plan_removals(root, [('a', 'f'), ('x', 'z')])
    assert result['removed_pairs'] == []
    assert result['stages'][1]['cycles_before'] == 0


def test_long_short_edge_cycle_is_removed_without_local_size_limit():
    root = network({name: 12 for name in 'abcdef'},
        [(a, b, 'l') for a, b in zip('abcdef', 'bcdefa')])
    result = plan_removals(root, [('a', 'f')])
    assert len(result['removed_pairs']) == 1
    assert result['removed_pairs'][0]['source'] == 'f'
    assert result['removed_pairs'][0]['target'] == 'a'
    assert not unreachable_od(apply_plan(root, result), [('a', 'f')])


def test_consecutive_removals_validate_current_graph_not_original_graph():
    # Two disjoint short cycles offer alternative a->z branches. Either first
    # turn can be removed, but removing both first turns would sever the OD.
    root = network(dict(a=100, z=100, b=12, c=12, d=12, e=12),
        [('a', 'b', 's'), ('b', 'c', 't'), ('c', 'b', 's'), ('c', 'z', 's'),
         ('a', 'd', 's'), ('d', 'e', 't'), ('e', 'd', 's'), ('e', 'z', 's')])
    result = plan_removals(root, [('a', 'z')])
    removed = {(row['source'], row['target']) for row in result['removed_pairs']}
    assert ('b', 'c') in removed
    assert ('d', 'e') not in removed
    assert ('e', 'd') in removed
    assert not unreachable_od(apply_plan(root, result), [('a', 'z')])
    assert all(stage['cycles_remaining'] == 0 for stage in result['stages'])


def test_signal_signature_tracks_phase_indications_by_connection_not_link_index():
    root = ET.fromstring('''<net><tlLogic id="t" programID="0" offset="2" type="static">
      <phase duration="10" state="Gr"/><phase duration="20" state="rG"/>
      </tlLogic><connection from="a" to="b" fromLane="0" toLane="0" tl="t" linkIndex="0"/>
      <connection from="c" to="d" fromLane="0" toLane="0" tl="t" linkIndex="1"/></net>''')
    programs, indications = signal_signature(root)
    assert indications[(('a', 'b', '0', '0'), 't', '0')] == ('G', 'r')
    assert programs['t', '0'][1] == [{'duration': '10'}, {'duration': '20'}]
    assert programs['t', '0'][0]['offset'] == '2'


def test_unreachable_input_is_rejected_before_any_removal():
    root = network(dict(a=12, b=12), [])
    with pytest.raises(ValueError, match='input contains unreachable'):
        plan_removals(root, [('a', 'b')])


def extra(source, target):
    return dict(source=source, target=target, reason='Observed blocking ring',
                evidence={'probe': 'seed_43', 'waiting_seconds': 600})


def test_extra_removal_breaks_observed_long_loop_and_preserves_all_active_od():
    root = network(dict(a=100, b=100, c=100, destination=100),
        [('a', 'b', 's'), ('b', 'c', 's'), ('c', 'a', 't'), ('c', 'destination', 's')])
    od = [('a', 'destination'), ('b', 'destination'), ('c', 'destination')]
    base = plan_removals(root, od)
    assert not base['removed_pairs']
    result = apply_extra_removals(root, od, base, [extra('c', 'a')])
    assert len(result['removed_pairs']) == 1
    assert result['extra_removals'][0]['status'] == 'removed'
    assert result['extra_removals'][0]['reason'] == 'Observed blocking ring'
    assert not unreachable_od(apply_plan(root, result), od)
    assert base['removed_pairs'] == []


def test_extra_removal_rejects_od_disconnection_without_mutating_base_plan():
    root = network(dict(a=100, b=100, c=100), [('a', 'b', 's'), ('b', 'c', 's')])
    od = [('a', 'c')]
    base = plan_removals(root, od)
    with pytest.raises(ValueError, match='disconnects active OD'):
        apply_extra_removals(root, od, base, [extra('a', 'b')])
    assert base['removed_pairs'] == []
    assert 'extra_removals' not in base


def test_extra_duplicate_of_base_removal_is_recorded_without_second_deletion():
    root = network(dict(a=12, b=12, c=12),
        [('a', 'b', 'l'), ('b', 'c', 's'), ('c', 'a', 't')])
    od = [('a', 'c')]
    base = plan_removals(root, od)
    assert [(row['source'], row['target']) for row in base['removed_pairs']] == [('c', 'a')]
    result = apply_extra_removals(root, od, base, [extra('c', 'a'), extra('c', 'a')])
    assert len(result['removed_pairs']) == 1
    assert [row['status'] for row in result['extra_removals']] == ['already_resolved', 'already_resolved']


def test_extras_validate_cumulative_connectivity_not_independent_original_graph():
    root = network(dict(a=100, b=100, c=100, destination=100),
        [('a', 'b', 's'), ('b', 'destination', 's'),
         ('a', 'c', 's'), ('c', 'destination', 's')])
    od = [('a', 'destination')]
    base = plan_removals(root, od)
    with pytest.raises(ValueError, match='disconnects active OD'):
        apply_extra_removals(root, od, base, [extra('a', 'b'), extra('a', 'c')])
    assert base['removed_pairs'] == []


def test_extra_requires_real_passenger_connection_and_documented_evidence():
    root = network(dict(a=100, b=100, c=100), [('a', 'b', 's'), ('b', 'c', 's')])
    base = plan_removals(root, [('a', 'c')])
    with pytest.raises(ValueError, match='not an actual passenger connection'):
        apply_extra_removals(root, [('a', 'c')], base, [extra('a', 'c')])
    with pytest.raises(ValueError, match='requires source, target, reason and evidence'):
        apply_extra_removals(root, [('a', 'c')], base, [dict(source='a', target='b')])


@pytest.mark.parametrize('custom_source', [False, True])
def test_cli_reads_selected_input_root_without_changing_sources(tmp_path, monkeypatch, custom_source):
    from tools import patch_short_road_cycles as patch
    monkeypatch.setattr(patch, 'ROOT', tmp_path / 'project')
    source_root = tmp_path / 'preserved_original_inputs' if custom_source else patch.ROOT / 'networks'
    city = 'coulommiers'
    source = source_root / city
    source.mkdir(parents=True)
    root = network(dict(a=100, b=100), [('a', 'b', 's')])
    for index, edge in enumerate(root.findall('edge')):
        edge.set('from', f'junction_{index}')
        edge.set('to', f'junction_{index + 1}')
        edge.find('lane').set('speed', '10')
    ET.ElementTree(root).write(source / f'{city}.net.xml', encoding='utf-8')
    (source / 'edge_registry.json').write_text('{}\n')
    (source / f'od_{city}.txt').write_text("{'origins': ['a'], 'destinations': ['b']}\n")
    (source / 'agents.csv').write_text('id,origin,destination,start_time\n0,0,0,0\n')
    original_bytes = {path.name: path.read_bytes() for path in source.iterdir()}
    output = tmp_path / 'prepared_revision'

    def copy_conversion(original, destination, work, netconvert, removed_pairs):
        # This test exercises source selection and real preparation/OD auditing;
        # netconvert conversion itself is independently validated on city maps.
        assert original.parent == output / 'original_inputs' / city
        assert removed_pairs == []
        shutil.copy2(original, destination)
        return {'removed_lane_connections': 0}

    monkeypatch.setattr(patch, 'build_patch', copy_conversion)
    argv = ['patch_short_road_cycles.py', '--cities', city, '--output-dir', str(output)]
    if custom_source:
        argv.extend(['--input-root', str(source_root)])
        assert not (patch.ROOT / 'networks').exists()
    monkeypatch.setattr(sys, 'argv', argv)
    patch.main()
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['input_root'] == str(source_root.resolve())
    assert manifest['networks'][city]['expected_vehicles'] == 1
    assert {path.name: path.read_bytes() for path in source.iterdir()} == original_bytes
    for name in original_bytes:
        assert (output / 'original_inputs' / city / name).read_bytes() == original_bytes[name]
    for name in ('agents.csv', f'od_{city}.txt'):
        assert (output / 'prepared' / city / name).read_bytes() == original_bytes[name]


def signal_network(phases, *, offset='2', control='static', phase_extras=None):
    root = ET.Element('net')
    logic = ET.SubElement(root, 'tlLogic', id='t', programID='0', offset=offset, type=control)
    for duration, state in phases:
        ET.SubElement(logic, 'phase', duration=str(duration), state=state, **(phase_extras or {}))
    ET.SubElement(root, 'connection', **{'from': 'a', 'to': 'b', 'fromLane': '0',
        'toLane': '0', 'tl': 't', 'linkIndex': '0'})
    return root


def test_semantically_identical_static_phase_compaction_is_accepted():
    old = signal_network([(40, 'G'), (5, 'r'), (40, 'G'), (5, 'G')])
    new = signal_network([(40, 'G'), (5, 'r'), (45, 'G')])
    result = validate_signal_equivalence(old, new, {('a', 'b', '0', '0')})
    assert result['equivalent_static_phase_compactions'] == [['t', '0']]
    assert result['retained_signal_connections_checked'] == 1


def test_one_second_green_change_is_rejected_even_with_equal_cycle_length():
    old = signal_network([(40, 'G'), (5, 'r'), (40, 'G'), (5, 'G')])
    new = signal_network([(41, 'G'), (4, 'r'), (45, 'G')])
    with pytest.raises(ValueError, match='changed signal timing or indications'):
        validate_signal_equivalence(old, new, {('a', 'b', '0', '0')})


def test_phase_compaction_does_not_allow_offset_change_or_cycle_rotation():
    old = signal_network([(40, 'G'), (5, 'r'), (40, 'G'), (5, 'G')])
    offset_changed = signal_network([(40, 'G'), (5, 'r'), (45, 'G')], offset='3')
    with pytest.raises(ValueError, match='offset'):
        validate_signal_equivalence(old, offset_changed, {('a', 'b', '0', '0')})
    rotated = signal_network([(85, 'G'), (5, 'r')])
    with pytest.raises(ValueError, match='changed signal timing or indications'):
        validate_signal_equivalence(old, rotated, {('a', 'b', '0', '0')})


@pytest.mark.parametrize('phase_extras, control', [({'name': 'hold'}, 'static'), ({}, 'actuated')])
def test_compaction_is_conservatively_rejected_with_extra_attrs_or_nonstatic_control(phase_extras, control):
    old = signal_network([(40, 'G'), (5, 'r'), (40, 'G'), (5, 'G')],
                         phase_extras=phase_extras, control=control)
    new = signal_network([(40, 'G'), (5, 'r'), (45, 'G')],
                         phase_extras=phase_extras, control=control)
    with pytest.raises(ValueError, match='non-static control or extra attributes'):
        validate_signal_equivalence(old, new, {('a', 'b', '0', '0')})


def test_static_signal_compaction_preserves_exact_decimal_durations():
    old = signal_network([('0.1', 'G'), ('0.2', 'G'), ('0.1', 'y'), ('0.2', 'r')])
    new = signal_network([('0.3', 'G'), ('0.1', 'y'), ('0.2', 'r')])
    validate_signal_equivalence(old, new, {('a', 'b', '0', '0')})
    new.find('tlLogic/phase').set('state', 'g')
    with pytest.raises(ValueError, match='changed signal timing or indications'):
        validate_signal_equivalence(old, new, {('a', 'b', '0', '0')})
