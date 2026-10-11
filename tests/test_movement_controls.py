from dataclasses import FrozenInstanceError
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from urban_dreamer.traffic.movement_controls import CityControlTable, UnsupportedControl


ROOT = Path(__file__).resolve().parents[1]


def fixture(tmp_path, *, junction_type='traffic_light', program=True, state='o'):
    # Shared TLS index 2 deliberately differs from junction request index 0.
    # The via lane precedes an internal waiting junction and is absent from
    # intLanes; mapping requests through intLanes.index(via) would fail.
    text = '''<net version="1.20">
      <edge id="a" from="A" to="J"><lane id="a_0" index="0" speed="10" length="20" shape="0,0 20,0"/>
        <lane id="a_1" index="1" speed="10" length="20" shape="0,0 20,0"/></edge>
      <edge id="b" from="B" to="J"><lane id="b_0" index="0" speed="10" length="20" shape="0,0 20,0"/></edge>
      <edge id="c" from="J" to="C"><lane id="c_0" index="0" speed="10" length="20" shape="20,0 40,0"/></edge>
      <edge id="d" from="J" to="D"><lane id="d_0" index="0" speed="10" length="20" shape="20,0 40,0"/></edge>
      <edge id=":J_0" function="internal"><lane id=":J_0_0" index="0" speed="10" length="2" shape="20,0 22,0"/></edge>
      <edge id=":J_3" function="internal"><lane id=":J_3_0" index="0" speed="10" length="2" shape="22,0 24,0"/></edge>
      PROGRAM
      <junction id="J" type="TYPE" x="20" y="0" incLanes="b_0 a_0 a_1" intLanes=":J_3_0">
        <request index="0" response="010" foes="110" cont="1"/>
        <request index="1" response="000" foes="101" cont="0"/>
        <request index="2" response="001" foes="011" cont="0"/>
      </junction>
      <junction id=":J_3_0" type="internal" x="22" y="0" incLanes=":J_0_0" intLanes=""/>
      <connection from="a" to="c" fromLane="1" toLane="0" tl="joined" linkIndex="1" dir="s" state="STATE"/>
      <connection from="a" to="c" fromLane="0" toLane="0" tl="joined" linkIndex="0" dir="s" state="STATE"/>
      <connection from="b" to="d" fromLane="0" toLane="0" via=":J_0_0" tl="joined" linkIndex="2" dir="l" state="STATE"/>
      <connection from=":J_0" to="d" fromLane="0" toLane="0" via=":J_3_0" dir="l" state="M"/>
      <connection from=":J_3" to="d" fromLane="0" toLane="0" dir="l" state="M"/>
    </net>'''
    logic = ('<tlLogic id="joined" type="static" programID="0" offset="1">'
             '<phase duration="2" state="Grg"/><phase duration="3" state="ryG"/></tlLogic>')
    path = tmp_path/'network.xml'
    path.write_text(text.replace('PROGRAM', logic if program else '').replace('TYPE', junction_type).replace('STATE', state))
    return path


def load(path, **kwargs):
    return CityControlTable.from_network(path, ['a', 'b', 'c', 'd'], no_external_overrides=True, **kwargs)


def test_request_indices_bits_and_lane_zero_representative(tmp_path):
    table = load(fixture(tmp_path))
    a, b = table.movements[(0, 2)], table.movements[(1, 3)]
    assert (a.from_lane, a.request_index, a.tls_index) == (0, 1, 0)
    assert (b.request_index, b.tls_index) == (0, 2)
    assert b.yields_to == frozenset({1})
    assert b.conflicts_with == frozenset({1, 2})
    assert table.preflight()['request_indices_validated'] == 3
    with pytest.raises(FrozenInstanceError):
        b.request_index = 2
    with pytest.raises(TypeError):
        table.movements[(0, 2)] = b


def test_static_offset_cycle_boundaries_and_conditional_yellow(tmp_path):
    table = load(fixture(tmp_path))
    assert [table.state((1, 3), tick) for tick in range(8)] == list('GggGGGgg')
    assert table.state((1, 3), 2.999) == 'g'
    assert table.state((1, 3), 3.) == 'G'
    assert table.classification((1, 3), 1.) == 'conditional'
    assert table.classification((0, 2), 0.) == 'blocked'
    # Select lane 1 only: its TLS index 1 gives yellow during the second phase.
    path = fixture(tmp_path)
    root = ET.parse(path).getroot()
    for c in root.findall('connection'):
        if c.get('from') == 'a':
            c.set('linkIndex', '1')
    ET.ElementTree(root).write(path)
    table = load(path)
    assert table.state((0, 2), 0.) == 'y'
    assert table.classification((0, 2), 0.) == 'conditional'


def test_missing_control_never_falls_back_to_green(tmp_path):
    path = fixture(tmp_path, program=False)
    with pytest.raises(UnsupportedControl, match='overrides'):
        CityControlTable.from_network(path, ['a', 'b', 'c', 'd'])
    table = load(path)
    assert table.preflight()['support_counts'] == {'unsupported': 2}
    with pytest.raises(UnsupportedControl, match='no static'):
        table.state((0, 2), 1.)
    with pytest.raises(UnsupportedControl, match='unregistered'):
        table.state((0, 3), 1.)
    with pytest.raises(ValueError, match='finite'):
        table.state((0, 2), float('nan'))


def test_rail_opt_in_is_explicitly_approximate(tmp_path):
    path = fixture(tmp_path, program=False, junction_type='rail_crossing')
    table = load(path)
    with pytest.raises(UnsupportedControl, match='rail crossing'):
        table.state((0, 2), 10.)
    approximate = load(path, verified_road_only_no_trains=True)
    assert approximate.preflight()['support_counts'] == {'road_only_rail_approximation': 2}
    assert approximate.state((0, 2), 10.) == 'G'


@pytest.mark.parametrize('junction_type,state,classification', [
    ('priority', 'M', 'protected'), ('priority', 'm', 'yield'),
    ('right_before_left', '=', 'yield'), ('zipper', 'Z', 'zipper'),
    ('traffic_light', 'm', 'yield')])
def test_uncontrolled_connection_classification(tmp_path, junction_type, state, classification):
    path = fixture(tmp_path, program=False, junction_type=junction_type, state=state)
    root = ET.parse(path).getroot()
    for c in root.findall('connection'):
        c.attrib.pop('tl', None)
        c.attrib.pop('linkIndex', None)
    ET.ElementTree(root).write(path)
    table = load(path)
    assert table.classification((0, 2), 1.) == classification


@pytest.mark.parametrize('mutation', ['response', 'index', 'phase', 'tls_index'])
def test_invalid_structure_or_program_is_explicit(tmp_path, mutation):
    path = fixture(tmp_path)
    root = ET.parse(path).getroot()
    if mutation == 'response':
        root.find('junction/request').set('response', '1000')
    elif mutation == 'index':
        root.find('junction/request').set('index', '4')
    elif mutation == 'phase':
        root.find('tlLogic/phase').set('duration', '2.5')
    else:
        for c in root.findall('connection'):
            if c.get('from') == 'a':
                c.set('linkIndex', '99')
    ET.ElementTree(root).write(path)
    if mutation in {'response', 'index'}:
        with pytest.raises(ValueError):
            load(path)
    else:
        with pytest.raises(UnsupportedControl):
            load(path).state((0, 2), 1.)


@pytest.mark.parametrize('city,pairs,unsupported', [
    ('ingolstadt_custom', 1426, 0), ('coulommiers', 2884, 2),
    ('montereau_fault_yonne', 3373, 12)])
def test_real_city_xml_matches_sumolib_indices(city, pairs, unsupported):
    import sumolib

    path = ROOT/'networks'/city/f'{city}.net.xml'
    root = ET.parse(path).getroot()
    ids = [edge.attrib['id'] for edge in root.findall('edge') if edge.get('function', 'normal') == 'normal']
    table = CityControlTable.from_network(path, ids, no_external_overrides=True)
    net = sumolib.net.readNet(str(path), withInternal=True, withPedestrianConnections=True)
    for movement in table.movements.values():
        lane = net.getEdge(ids[movement.source_edge]).getLane(movement.from_lane)
        connection = next(c for c in lane.getOutgoing() if c.getTo().getID() == ids[movement.target_edge])
        assert connection.getJunctionIndex() == movement.request_index
        node = connection.getFrom().getToNode()
        assert frozenset(i for i, bit in enumerate(reversed(node._prohibits[movement.request_index])) if bit == '1') == movement.yields_to
    report = table.preflight()
    assert report['movement_count'] == pairs
    assert report['support_counts'].get('unsupported', 0) == unsupported
