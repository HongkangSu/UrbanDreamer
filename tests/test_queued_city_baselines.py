"""Resource admission must reserve memory before CUDA allocates it."""

import json
from types import SimpleNamespace

import pytest

from tools import queued_city_baselines as queue


def test_hot_admission_overrides_release_capacity_and_keep_peak_and_oom_floors(tmp_path):
    entries = {key:dict(required_gpu_mib=value) for key,value in [('small',2048),('peak',2048),('retry',4096)]}
    jobs = {key:dict(required_gpu_mib=e['required_gpu_mib'], observed_peak_mib=peak, oom_retries=retry)
            for key,e,peak,retry in [('small',entries['small'],404,0),('peak',entries['peak'],1800,0),('retry',entries['retry'],360,1)]}
    manifest = dict(gpu_headroom_mib=1024,poll_seconds=10)
    queue.write_json(tmp_path/'admission_overrides.json',dict(gpu_headroom_mib=512,poll_seconds=5,
        peak_margin_fraction=.1,peak_padding_mib=128,required_gpu_mib=dict(small=768,peak=768,retry=768)))
    assert queue.apply_admission_overrides(tmp_path,manifest,entries,jobs)
    assert jobs['small']['required_gpu_mib']==768
    assert jobs['peak']['required_gpu_mib']==2304
    assert jobs['retry']['required_gpu_mib']==4096
    assert manifest['gpu_headroom_mib']==512 and manifest['poll_seconds']==5
    policy=json.loads((tmp_path/'admission_overrides.json').read_text())
    policy['required_gpu_mib']['small']=1024
    queue.write_json(tmp_path/'admission_overrides.json',policy)
    queue.apply_admission_overrides(tmp_path,manifest,entries,jobs)
    assert jobs['small']['required_gpu_mib']==1024


def test_invalid_hot_budget_does_not_partially_change_admission(tmp_path):
    manifest = dict(gpu_headroom_mib=1024,poll_seconds=10)
    queue.write_json(tmp_path/'admission_overrides.json',dict(gpu_headroom_mib=512,required_gpu_mib=dict(unknown=768)))
    with pytest.raises(ValueError,match='invalid admission'):
        queue.apply_admission_overrides(tmp_path,manifest,{}, {})
    assert manifest['gpu_headroom_mib']==1024


def test_user_stopped_batch_cannot_auto_resume(tmp_path, monkeypatch):
    (tmp_path / 'manifest.json').write_text(json.dumps({'launch_status': 'stopped'}))
    monkeypatch.setattr(queue.subprocess, 'Popen', lambda *a, **kw: pytest.fail('stopped batch launched a worker'))
    with pytest.raises(ValueError, match='batch stopped by user'):
        queue.run(tmp_path)
    assert not (tmp_path / 'queue.lock').exists()
    assert not (tmp_path / 'queue_state.json').exists()


@pytest.mark.parametrize('incident_start', [1e6, 5000])
def test_prepare_disables_prior_and_checks_actual_deadline(tmp_path, monkeypatch, incident_start):
    from tools import run_city_baseline_dataset as dataset
    from urban_dreamer.workflow import config as configuration
    from urban_dreamer.traffic import routing

    def snapshot(root):
        source = root / 'source'
        source.mkdir()
        (source / 'networks').symlink_to(tmp_path, target_is_directory=True)
        return source

    def load_config(path):
        return SimpleNamespace(city=path.stem,
            simulation=SimpleNamespace(traffic_model='meso', incident_start_seconds=incident_start,
                world_model_intervals_per_episode=120, training_completion_buffer_seconds=2000,
                evaluation_horizon_seconds=5600))

    def validate(cfg):
        return SimpleNamespace(input_hashes={}, vehicle_count={
            'ingolstadt_custom': 1035, 'coulommiers': 1811, 'montereau_fault_yonne': 1500}[cfg.city])

    def build_runs(root, source, settings, stamp):
        assert settings.sp_prior_weight == 0.
        return [dict(id='ingolstadt_custom_' + method, city='ingolstadt_custom', method=method,
            command=['trainer', '--device', 'cpu', '--sp-prior-weight', str(settings.sp_prior_weight)])
            for method in queue.METHODS]

    monkeypatch.setattr(dataset, 'snapshot', snapshot)
    monkeypatch.setattr(dataset, 'build_runs', build_runs)
    monkeypatch.setattr(configuration, 'load_config', load_config)
    monkeypatch.setattr(routing, 'validate_prepared_inputs', validate)
    monkeypatch.setattr(queue.shutil, 'copy2', lambda *a: None)
    output = tmp_path / 'batch'
    settings = SimpleNamespace(output_root=output, min_free_gb=0,
        episodes=5000, poll_seconds=10, gpu_headroom_mib=1024)
    if incident_start <= 5610:
        with pytest.raises(AssertionError):
            queue.prepare(settings)
        return
    queue.prepare(settings)
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['prior']['initial_weight'] == 0.
    assert all(entry['effective_prior'] == 0. for entry in manifest['runs'])
    for entry in manifest['runs']:
        command = entry['command']
        if '--sp-prior-weight' in command:
            assert float(command[command.index('--sp-prior-weight') + 1]) == 0.


def test_starting_jobs_reserve_memory_before_cuda_allocation():
    gpu = {"index": 0, "free_mib": 10_000}
    running = [{"gpu_index": 0, "required_gpu_mib": 6_000, "pid": 101}]
    assert queue.effective_gpu_free_mib(gpu, running, {}) == 2_976
    assert queue.choose_gpu(3_000, [gpu], running, {}) is None


def test_realized_cuda_memory_is_not_reserved_twice():
    gpu = {"index": 0, "free_mib": 10_000}
    running = [{"gpu_index": 0, "required_gpu_mib": 6_000, "pid": 101}]
    assert queue.effective_gpu_free_mib(gpu, running, {101: 4_000}) == 6_976
    assert queue.effective_gpu_free_mib(gpu, running, {101: 7_000}) == 8_976


def test_gpu_reservations_follow_device_index():
    gpus = [{"index": 2, "free_mib": 9_000}, {"index": 7, "free_mib": 8_000}]
    running = [{"gpu_index": 2, "required_gpu_mib": 6_000, "pid": 101}]
    assert queue.choose_gpu(4_000, gpus, running, {}) == 7


def test_small_job_can_backfill_while_larger_job_waits():
    gpus = [{"index": 0, "free_mib": 8_000}]
    running = [{"gpu_index": 0, "required_gpu_mib": 4_000, "pid": 101}]
    assert queue.choose_gpu(4_000, gpus, running, {}) is None
    assert queue.choose_gpu(2_000, gpus, running, {}) == 0


def test_gpu_headroom_is_respected_at_exact_boundary():
    gpus = [{"index": 0, "free_mib": 5_024}]
    assert queue.choose_gpu(4_000, gpus, [], {}) == 0
    assert queue.choose_gpu(4_001, gpus, [], {}) is None


@pytest.mark.parametrize("log", [
    "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
    "RuntimeError: CUDA out of memory",
    "CUDA error: out of memory",
])
def test_cuda_oom_is_recognized_for_retry(log):
    assert queue.failure_kind(log, 1) == "oom"


@pytest.mark.parametrize("log", [
    "FileNotFoundError: agents.csv is missing",
    "RuntimeError: SUMO connection closed",
    "CUDA error: device-side assert triggered",
    "An exception occurred while loading out of memory test fixture",
])
def test_unrelated_failures_are_not_treated_as_cuda_oom(log):
    assert queue.failure_kind(log, 1) == "failed"


def test_oom_requirement_increases_and_never_exceeds_usable_capacity():
    assert queue.bump_gpu_requirement(4_000, 3_000, 12_000) == 6_000
    assert queue.bump_gpu_requirement(4_000, 5_000, 12_000) >= 6_000
    assert queue.bump_gpu_requirement(10_000, 9_000, 12_000) == 12_000


def test_mature_small_worker_releases_excess_initial_reservation():
    assert queue.steady_gpu_requirement('mappo', 568, 3072, 98) == 1024


def test_observed_peak_raises_underestimated_budget():
    assert queue.steady_gpu_requirement('gnac', 6176, 4096, 1100) == 7424


@pytest.mark.parametrize('episode,warmup', [(49, True), (400, False)])
def test_startup_and_model_warmup_keep_initial_budget(episode, warmup):
    assert queue.steady_gpu_requirement('mamba', 760, 4096, episode,
                                        warmup_complete=warmup) == 4096


def test_oom_retry_budget_is_never_reduced_by_calibration():
    assert queue.steady_gpu_requirement('iql', 362, 4096, 500, oom_retries=1) == 4096


def test_peer_calibration_keeps_retry_budget(monkeypatch):
    entries = {
        'coulommiers_iql': {'city': 'coulommiers', 'method': 'iql'},
        'montereau_fault_yonne_iql': {'city': 'montereau_fault_yonne', 'method': 'iql'},
    }
    peer = dict(status='running', required_gpu_mib=2048, observed_peak_mib=362,
                oom_retries=0, attempt=1)
    waiting = dict(status='waiting_gpu', required_gpu_mib=2048, observed_peak_mib=0,
                   oom_retries=0, attempt=0)
    monkeypatch.setattr(queue, 'durable_episode', lambda entry: 68)
    jobs = {'coulommiers_iql': peer, 'montereau_fault_yonne_iql': waiting}
    queue.calibrate_memory_budgets(entries, jobs, [], {})
    assert waiting['required_gpu_mib'] == 1280
    waiting.update(required_gpu_mib=4096, oom_retries=1)
    queue.calibrate_memory_budgets(entries, jobs, [], {})
    assert waiting['required_gpu_mib'] == 4096


@pytest.mark.parametrize('wait_status,expected', [(0, 0), (256, 1), (15, -15)])
def test_adopted_worker_exit_preserves_success_failure_and_signal(monkeypatch, wait_status, expected):
    fields = ['0'] * 50
    fields[0], fields[19], fields[49] = 'Z', '12345', str(wait_status)
    monkeypatch.setattr(queue.Path, 'read_text', lambda self: '101 (python) ' + ' '.join(fields))
    assert queue.adopted_exit_code({'pid':101, 'process_starttime':'12345'}) == expected
    assert queue.adopted_exit_code({'pid':101, 'process_starttime':'other'}) is None


def test_cpu_job_finishes_when_gpu_query_is_unavailable(tmp_path, monkeypatch):
    """An unavailable GPU must not prevent the independent CPU queue."""
    source = tmp_path / "source"
    source.mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "processes").mkdir()
    entry = {
        "id": "city_random", "city": "city", "method": "random", "device": "cpu",
        "episode_budget": 1, "required_gpu_mib": 0, "required_host_mib": 1024,
        "output_dir": str(tmp_path / "city" / "random"),
        "log": str(tmp_path / "logs" / "random.log"),
        "state": str(tmp_path / "processes" / "random.json"),
        "command": ["python", "-u", "collect.py", "--device", "cpu"],
    }
    manifest = {
        "source_snapshot": str(source), "runs": [entry], "reward_mode": "global_with_individual_timeout",
        "gpu_headroom_mib": 1024, "host_headroom_mib": 1024, "min_free_gb": 0.001,
        "max_cpu_jobs": 12, "max_oom_retries": 12, "poll_seconds": 0.001,
    }
    state = {"status": "queued", "pid": None, "attempt": 0, "required_gpu_mib": 0,
             "observed_peak_mib": 0, "oom_retries": 0, "next_eligible": 0}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "queue_state.json").write_text(json.dumps({"jobs": {entry["id"]: state}}))
    (tmp_path / "source_hashes.json").write_text("{}")
    launches = []

    class FinishedProcess:
        pid = 101
        returncode = None

        def poll(self):
            self.returncode = 0
            return 0

    def launch(command, **kwargs):
        launches.append((command, kwargs))
        output = tmp_path / "city" / "random" / "episodes" / "train"
        output.mkdir(parents=True)
        (output / "episode_1.npz").touch()
        return FinishedProcess()

    def gpu_unavailable():
        raise OSError("nvidia-smi unavailable")

    monkeypatch.setattr(queue, "query_gpus", gpu_unavailable)
    monkeypatch.setattr(queue, "choose_gpu", lambda *a, **kw: pytest.fail("CPU job requested a GPU"))
    monkeypatch.setattr(queue, "host_available_mib", lambda: 65536)
    monkeypatch.setattr(queue, "process_identity", lambda pid: "12345")
    monkeypatch.setattr(queue, "durable_episode", lambda entry: 1)
    monkeypatch.setattr(queue.subprocess, "Popen", launch)
    monkeypatch.setattr(queue.time, "sleep", lambda duration: None)
    queue.run(tmp_path)

    assert len(launches) == 1
    assert launches[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    result = json.loads((tmp_path / "final_status.json").read_text())
    assert result["counts"] == {"completed": 1}
    assert result["gpu_query_error"] == "nvidia-smi unavailable"
def test_default_methods_exclude_dta_and_random():
    from tools import queued_city_baselines as queue
    assert queue.METHODS == queue.LEARNED
    assert len(queue.METHODS) == 8

