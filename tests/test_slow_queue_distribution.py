import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('queue_audit', Path(__file__).parents[1]/'tools/audit_slow_queue_distribution.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_two_boundary_speed_and_persistent_masks():
    flow = np.zeros((3, 2, 8), dtype=np.float32)
    flow[:, :, 2] = .5
    flow[:, :, 1] = [[0, .1], [.3, .1], [0, .1]]
    count, slow, persistent, localized, widespread = audit.queue_masks(flow, np.array([10, 10]), np.array([10, 10]))
    assert slow.all()
    assert not persistent[:, 0].any()
    assert persistent[:, 1].all()
    assert localized.all()
    assert not widespread.any()


def test_summary_keeps_denominators_and_zero_windows():
    flow = np.zeros((2, 6, 8), dtype=np.float32)
    flow[0, 0, 2] = 5
    row = audit.summarize_flow(flow, np.ones(6), np.ones(6), [0])
    assert row['totals']['total_road_cells'] == 6
    assert row['totals']['slow_cells'] == 1
    assert row['totals']['occupied_cells'] == 1
    assert row['critical_edges']['0']['slow_count_5_9'] == 1
    empty = audit.summarize_flow(flow[:1], np.ones(6), np.ones(6))
    assert empty['totals']['windows'] == 0


def test_sampler_coverage_counts_partial_episode():
    checkpoint = dict(sampler=dict(epoch=2, order=[1, 0], cursor=1, frame_cursor=3),
                      update=9, batch_size=4)
    stats = audit.sampler_coverage(checkpoint, [dict(frames=10), dict(frames=20)])
    assert stats['total_raw_frames_seen'] == 83
    assert stats['current_epoch_frames_seen'] == 23


def test_float_count_threshold_and_widespread_context():
    flow = np.zeros((2, 21, 8), dtype=np.float32)
    flow[:, :, 2] = 4.999999
    stats = audit.summarize_flow(flow, np.ones(21), np.ones(21))['totals']
    assert stats['slow_cells'] == 21
    assert stats['widespread_slow_windows'] == 1
    assert stats['localized_slow_cells'] == 0
