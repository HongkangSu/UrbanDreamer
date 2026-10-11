"""Pure phase bookkeeping and arithmetic tests for the isolated runtime profiler."""
import pytest
from tools.profile_imagination_runtime import PhaseRecorder, summarize_phases, summarize_trace, velocity_matrix_flops


def test_phase_records_failure_and_preserves_exception():
    recorder = PhaseRecorder()
    with pytest.raises(RuntimeError, match='sentinel'):
        with recorder.phase('worker_wait', windows=3):
            raise RuntimeError('sentinel')
    rows = recorder.finish()
    assert len(rows) == 1
    assert rows[0]['details'] == {'windows': 3}
    assert rows[0]['gpu_span_seconds'] is None
    assert rows[0]['wall_seconds'] >= 0
    assert recorder.finish() == []


def test_summary_does_not_double_count_different_phase_names():
    rows = [dict(phase='predict', wall_seconds=2., gpu_span_seconds=1.5),
            dict(phase='predict', wall_seconds=3., gpu_span_seconds=2.5),
            dict(phase='worker', wall_seconds=4., gpu_span_seconds=None)]
    summary = summarize_phases(rows)
    assert summary['predict'] == dict(calls=2, wall_seconds=5., gpu_span_seconds=4.)
    assert summary['worker']['wall_seconds'] == 4.


def test_gated_mlp_and_temporal_value_projection_are_counted():
    # Scalar width, two tokens: attention=16, projections=24, gated MLP=36.
    assert velocity_matrix_flops(edges=1, tokens=2, width=1, hidden=3,
                                 depth=1, latent_width=1) == 80
    with pytest.raises(ValueError):
        velocity_matrix_flops(edges=3, tokens=2, width=1, hidden=3, depth=1, latent_width=1)


def test_kernel_trace_separates_device_work_from_host_launches():
    result = summarize_trace([
        dict(ph='X', cat='kernel', name='gemm', dur=2000),
        dict(ph='X', cat='cuda_runtime', name='cudaLaunchKernel', dur=100),
        dict(ph='X', cat='cpu_op', name='aten::mm', dur=3000),
        dict(ph='M', cat='kernel', name='process_name', dur=9000)])
    assert result['kernel_seconds'] == .002
    assert result['cuda_runtime_host_seconds'] == .0001
    assert len(result['rows']) == 2
