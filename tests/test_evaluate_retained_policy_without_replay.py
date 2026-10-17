from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from tools.evaluate_retained_policy_without_replay import forbid_old_data, restore_models_only


def test_raw_guard_blocks_reads_and_stats(tmp_path):
    raw=tmp_path/'old.npz'; raw.write_bytes(b'raw')
    keep=tmp_path/'model.pt';keep.write_bytes(b'checkpoint')
    with forbid_old_data([raw]) as attempts:
        with pytest.raises(RuntimeError):raw.read_bytes()
        with pytest.raises(RuntimeError):raw.stat()
        assert keep.read_bytes()==b'checkpoint'
    assert len(attempts)==2


def test_restore_only_models_metrics_and_prior_schedule():
    model=Mock(); trainer=SimpleNamespace(actor=model, _set_online_prior=Mock(), optimizers=Mock(), online_replay=Mock())
    state={'models':{'actor':{'weight':1}},'metrics':{'online_episodes_completed':2000},'next_environment_episode':2000,
           'optimizers':object(),'replay':object(),'rng':object()}
    restore_models_only(trainer,state)
    model.load_state_dict.assert_called_once_with({'weight':1},strict=True)
    assert trainer._next_environment_episode==2000
    trainer._set_online_prior.assert_called_once_with()
    assert trainer.optimizers.mock_calls==[] and trainer.online_replay.mock_calls==[]
