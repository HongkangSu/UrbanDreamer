"""New corpus controls retain frozen membership and source-integrity semantics."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools import run_complete_pretrain as runner
from tools import train_codec_ready as codec


def test_cache_hashes_only_initial_access_and_stamp_change(tmp_path, monkeypatch):
    path = tmp_path / 'episode.npz'
    arrays = dict(flow=np.zeros((3, 2, 8), np.float32), summary_json=np.arange(5, dtype=np.uint8))
    np.savez_compressed(path, **arrays)
    entry = dict(path=str(path), flow_member_sha256=codec.member_hash(path, 'flow'),
                 metadata_member_sha256=codec.member_hash(path, 'summary_json'))
    calls = []
    original = codec.member_hash
    monkeypatch.setattr(codec, 'member_hash', lambda *a: (calls.append(a), original(*a))[1])
    cache = codec.FlowCache(2)
    cache.get(entry)
    cache.get(entry)
    assert len(calls) == 2
    np.savez(path, **arrays)
    assert np.array_equal(cache.get(entry), arrays['flow'])
    assert len(calls) == 4
    arrays['flow'][0, 0, 1] = 3
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='content changed'):
        cache.get(entry)


@pytest.mark.parametrize('index', [0, 1])
def test_codec_cap_applies_selected_gpu(monkeypatch, index):
    calls = []
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (8 * 1024**3, 24 * 1024**3))
    monkeypatch.setattr(torch.cuda, 'set_per_process_memory_fraction', lambda *args: calls.append(args))
    device, info = codec.choose_device(f'cuda:{index}', .2)
    assert device.index == index
    assert calls == [(.2, index)]
    assert info['memory_fraction'] == .2
    calls.clear()
    codec.choose_device(f'cuda:{index}')
    assert calls == [(.05, index)]
    assert codec.choose_device('cpu')[0].type == 'cpu'


@pytest.mark.parametrize('module, extra', [(codec, []), (runner, [
    '--config', 'unused', '--codec-dir', 'unused', '--cache-manifest', 'unused', '--output-dir', 'unused'])])
def test_invalid_expected_membership_rejected_before_io(module, extra):
    with pytest.raises(SystemExit):
        module.main(extra + ['--expected-train-episodes', '0'])


def test_persistent_queue_metric_is_endpoint_based_and_weighted():
    class IdentityCodec:
        def __call__(self, flow):
            return flow
        def decode(self, flow):
            decoded = flow.clone()
            decoded[..., 1] = .5
            return decoded
    network = SimpleNamespace(capacities=[10., 10.], speed_limits_mps=[10., 10.])
    flow = torch.zeros(1, 2, 8)
    flow[..., 1] = .1
    flow[..., 2] = .6
    future = flow.clone()
    future[:, 1, 1] = .4  # Only first road remains slow at the second endpoint.
    metrics = codec.validation_metrics(IdentityCodec(), flow, future, network)
    assert metrics['codec_persistent_slow_road_frames'] == 1
    assert metrics['codec_persistent_slow_speed_mae_mps'] == 4
    accumulator = codec.MetricAccumulator()
    accumulator.add(metrics, 1, 2)
    accumulator.add(metrics, 1, 2)
    result = accumulator.result()
    assert result['codec_persistent_slow_road_frames'] == 2
    assert result['codec_persistent_slow_speed_bias_mps'] == 4
    assert 'codec_persistent_slow_speed_mae_mps' not in codec.GATES


@pytest.mark.parametrize('flags, expected', [([], {}), ([
    '--expected-train-episodes', '121', '--expected-validation-episodes', '17'],
    {'expected_train': 121, 'expected_validation': 17})])
def test_complete_runner_forwards_explicit_counts_preserving_legacy_defaults(tmp_path, monkeypatch, flags, expected):
    simulation = object()
    monkeypatch.setattr(runner, 'load_config', lambda _: SimpleNamespace(simulation=simulation))
    class Captured(Exception):
        pass
    def prerequisites(*args, **kwargs):
        assert kwargs == dict(expected_simulation=simulation, **expected)
        raise Captured
    monkeypatch.setattr(runner, 'prerequisite_state', prerequisites)
    with pytest.raises(Captured):
        runner.main(['--config', 'unused', '--codec-dir', 'unused', '--cache-manifest', 'unused',
                     '--output-dir', str(tmp_path / 'output'), '--device', 'cuda:0'] + flags)
    assert not (tmp_path / 'output').exists()
