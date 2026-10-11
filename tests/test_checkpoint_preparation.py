"""Checkpoint preparation keeps replay snapshots independently mutable."""
from dataclasses import asdict, fields

import numpy as np
import pytest
import torch

from urban_dreamer.learning.action_events import ActionEvent
from urban_dreamer.learning.replay import MacroTransition
from urban_dreamer.workflow.checkpoint import CheckpointError, _normalize_plain_value


@pytest.mark.parametrize('events', [None, (), (ActionEvent('v', 0, 1., 0, 1, 1),)])
def test_transition_snapshot_retains_schema_and_ownership(events):
    row = MacroTransition(
        flow=torch.zeros(2, 8), next_flow=torch.ones(2, 8),
        action=torch.zeros(2), departure=torch.zeros(2), context=torch.zeros(2),
        active_vehicle_fraction=0.5,
        vehicles=({'vehicle_id': 'v', 'nested': {'route': [0, 1]}},),
        next_vehicles=({'vehicle_id': 'v', 'nested': {'route': [1]}},),
        action_events=events,
    )
    snapshot = row.state_dict()
    assert set(snapshot) == {field.name for field in fields(row)}
    assert snapshot['action_events'] == (None if events is None else [asdict(e) for e in events])
    restored = MacroTransition.from_state_dict(snapshot)
    assert restored.action_events == events
    snapshot['vehicles'][0]['nested']['route'].append(9)
    snapshot['next_vehicles'][0]['nested']['route'].append(9)
    snapshot['flow'].fill_(9)
    if events:
        snapshot['action_events'][0]['next_edge'] = 0
        assert row.action_events[0].next_edge == restored.action_events[0].next_edge == 1
    assert row.vehicles[0]['nested']['route'] == restored.vehicles[0]['nested']['route'] == [0, 1]
    assert row.next_vehicles[0]['nested']['route'] == restored.next_vehicles[0]['nested']['route'] == [1]
    assert not row.flow.any() and not restored.flow.any()


@pytest.mark.parametrize('value', [None, True, False, 'road', 2, 2.5, np.float32(2.5), np.int64(2), np.bool_(True)])
def test_scalar_normalization_remains_weights_only_safe(value, tmp_path):
    result = _normalize_plain_value(value, path='root')
    assert result == value
    assert result is None or type(result) in (str, float, int, bool)
    torch.save(result, tmp_path / 'value.pt')
    assert torch.load(tmp_path / 'value.pt', weights_only=True) == result


def test_normalization_preserves_container_conversion_and_error_path():
    original = {'route': ({'index': np.int64(1)},), 'state': torch.tensor([1.])}
    result = _normalize_plain_value(original, path='root')
    assert isinstance(result['route'], list)
    result['route'][0]['index'] = 7
    assert original['route'][0]['index'] == 1
    with pytest.raises(CheckpointError, match=r'root.bad.*object-dtype'):
        _normalize_plain_value({'bad': np.array([object()], dtype=object)}, path='root')
    with pytest.raises(CheckpointError, match='key'):
        _normalize_plain_value({True: 1}, path='root')
