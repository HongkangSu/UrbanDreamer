from copy import deepcopy

import numpy as np
import pytest
import torch

from tools.train_codec_ready import (FlowCache, FrameSampler, MetricAccumulator, SparseGraphCodec,
    cohort_codec_anchors, configured_speed_gate, gate_result, joint_gate_result, member_hash)
from urban_dreamer.workflow.config import CodecConfig
from urban_dreamer.models.world_model import FlowProjector


def test_sparse_codec_matches_dense_outputs_and_parameter_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(32)
    adjacency = (torch.rand(11, 11) > .8).float()
    adjacency /= adjacency.sum(-1, keepdim=True).clamp_min(1.)
    dense = FlowProjector(adjacency=adjacency, positive_speed=True)
    sparse = SparseGraphCodec(adjacency=adjacency, positive_speed=True)
    sparse.load_state_dict(dense.state_dict())
    assert set(sparse.state_dict()) == set(dense.state_dict())
    flow = torch.randn(2, 3, 11, 8)
    dense_output = dense.decode(dense(flow))
    sparse_output = sparse.decode(sparse(flow))
    assert torch.allclose(dense_output, sparse_output, atol=1e-6, rtol=1e-5)
    dense_output.square().mean().backward()
    sparse_output.square().mean().backward()
    for (left_name, left), (right_name, right) in zip(dense.named_parameters(), sparse.named_parameters()):
        assert left_name == right_name
        assert (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            assert torch.allclose(left.grad, right.grad, atol=1e-6, rtol=1e-4), left_name


def test_sampler_visits_every_frame_once_in_epoch_and_retains_partial_batches():
    entries = [dict(frames=5), dict(frames=7)]
    sampler = FrameSampler(entries, batch_size=3, seed=12)
    batches = [sampler.next() for _ in range(5)]
    seen = [(episode, int(frame)) for episode, frames in batches for frame in frames]
    assert len(seen) == len(set(seen)) == 12
    assert set(seen) == {(episode, frame) for episode, row in enumerate(entries) for frame in range(row['frames'])}
    assert sorted(len(frames) for _, frames in batches) == [1, 2, 3, 3, 3]
    sampler.next()
    assert sampler.epoch == 1


def test_sampler_resumes_mid_episode_with_identical_next_frame_batches():
    entries = [dict(frames=9), dict(frames=11)]
    original = FrameSampler(entries, batch_size=4, seed=42)
    original.next()
    restored = FrameSampler(entries, batch_size=4, seed=999)
    restored.load_state_dict(deepcopy(original.state_dict()))
    for _ in range(10):
        left_episode, left_frames = original.next()
        right_episode, right_frames = restored.next()
        assert left_episode == right_episode
        assert np.array_equal(left_frames, right_frames)


def test_metric_accumulator_weights_masks_and_frames_instead_of_batch_means():
    accumulator = MetricAccumulator()
    accumulator.add(dict(codec_count_mae_occupied=2., codec_count_mae_empty=.1, codec_count_mae_all=1.,
        codec_total_relative_error=.2, codec_speed_mae_mps=1., codec_count_occupied_roads=1.,
        codec_count_empty_roads=3., codec_count_rmse_all=2., codec_empty_predicted_total=.3), frames=2, edges=2)
    accumulator.add(dict(codec_count_mae_occupied=0., codec_count_mae_empty=.3, codec_count_mae_all=.1,
        codec_total_relative_error=.0, codec_speed_mae_mps=.1, codec_count_occupied_roads=0.,
        codec_count_empty_roads=2., codec_count_rmse_all=1., codec_empty_predicted_total=.6), frames=1, edges=2)
    result = accumulator.result()
    assert result['codec_count_mae_occupied'] == 2.
    assert result['codec_count_mae_empty'] == pytest.approx(.18)
    assert result['codec_count_mae_all'] == pytest.approx(.7)
    assert result['codec_total_relative_error'] == pytest.approx(.4 / 3)
    assert result['codec_count_rmse_all'] == pytest.approx(3 ** .5)
    assert result['codec_empty_predicted_total'] == pytest.approx(.9)
    assert result['codec_count_occupied_roads'] == 1.


def test_gate_never_passes_missing_occupied_validation_or_nonfinite_metric():
    metrics = dict(codec_count_mae_occupied=.1, codec_count_mae_all=.02, codec_count_mae_empty=.01,
        codec_total_relative_error=.05, codec_speed_mae_mps=.5)
    assert gate_result(metrics)['passed']
    metrics['codec_count_mae_occupied'] = None
    assert not gate_result(metrics)['passed']
    metrics['codec_count_mae_occupied'] = float('nan')
    assert not gate_result(metrics)['passed']


def test_slow_metrics_use_actual_mask_counts_across_batches():
    accumulator = MetricAccumulator()
    accumulator.add(dict(codec_count_occupied_roads=2., codec_count_empty_roads=6.,
        codec_slow_roads=1., codec_speed_mae_slow_mps=2.), frames=2, edges=4)
    accumulator.add(dict(codec_count_occupied_roads=4., codec_count_empty_roads=0.,
        codec_slow_roads=3., codec_speed_mae_slow_mps=0.), frames=1, edges=4)
    result = accumulator.result()
    assert result['codec_slow_roads'] == 4.
    assert result['codec_speed_mae_slow_mps'] == .5


def test_configured_gate_rejects_slow_failure_and_absent_slow_validation():
    limits = CodecConfig(slow_validation_min_roads=1)
    metrics = dict(codec_speed_mae_occupied_mps=.5, codec_speed_mae_slow_mps=.2,
                   codec_slow_roads=2)
    assert configured_speed_gate(metrics, limits)['passed']
    assert not configured_speed_gate(dict(metrics, codec_speed_mae_slow_mps=2.), limits)['passed']
    assert not configured_speed_gate(dict(metrics, codec_slow_roads=0), limits)['passed']


def test_joint_gate_requires_raw_and_hybrid_so_raw_ready_resume_keeps_training():
    raw = dict(codec_count_mae_occupied=.48, codec_count_mae_all=.06, codec_count_mae_empty=.004,
               codec_total_relative_error=.06, codec_speed_mae_mps=.2)
    hybrid = dict(raw, codec_total_relative_error=.68)
    assert gate_result(raw)['passed']
    assert not joint_gate_result(raw, hybrid)['passed']
    assert joint_gate_result(raw, raw)['passed']


def test_cohort_anchors_match_seq16_and_always_have_actual_next_boundary():
    starts, positions, length = cohort_codec_anchors(127)
    assert starts.tolist() == [0, 37, 74, 111]
    assert positions.tolist() == [0, 5, 10, 15]
    assert length == 16
    assert all(start + position + 1 <= 127 for start in starts for position in positions)
    starts, positions, length = cohort_codec_anchors(3)
    assert starts.tolist() == [0]
    assert positions.tolist() == [0, 1, 2]
    assert length == 3


def test_flow_cache_is_bounded_and_detects_content_change_despite_recompression(tmp_path):
    cache = FlowCache(2)
    entries = []
    flow = np.zeros((3, 2, 8), dtype=np.float32)
    for index in range(3):
        path = tmp_path / f'{index}.npz'
        np.savez_compressed(path, flow=flow + index, summary_json=np.arange(4, dtype=np.uint8))
        entry = dict(path=str(path), flow_member_sha256=member_hash(path, 'flow'),
                     metadata_member_sha256=member_hash(path, 'summary_json'))
        entries.append(entry)
        cache.get(entry)
        assert len(cache.cache) <= 2
    np.savez(entries[2]['path'], flow=flow + 2, summary_json=np.arange(4, dtype=np.uint8))
    assert cache.get(entries[2]).shape == flow.shape
    np.savez(entries[2]['path'], flow=flow + 9, summary_json=np.arange(4, dtype=np.uint8))
    with pytest.raises(ValueError, match='content changed'):
        cache.get(entries[2])
