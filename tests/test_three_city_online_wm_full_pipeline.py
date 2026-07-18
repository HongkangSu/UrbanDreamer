import hashlib
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_three_city_online_wm_full_pipeline as pipeline  # noqa: E402
from scripts.run_three_city_online_wm_full_pipeline import (  # noqa: E402
    CITY_SPECS,
    EXISTING_WARMUP_SPECS,
    _base_environment,
    build_arg_parser,
    build_collect_command,
    build_eval_command,
    build_online_command,
    build_wm_command,
    city_paths,
    prepare_city_inputs,
    resolve_warmup_paths,
    resolve_world_model_paths,
    run_city,
    run_cities,
    runtime_preflight,
)


def _value_after(command, flag):
    return command[command.index(flag) + 1]


def _write_fake_city_inputs(root: Path, city: str) -> None:
    spec = CITY_SPECS[city]
    settings = root / spec.settings_rel
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(
        '\n'.join(
            [
                '# generated settings',
                f'NETWORK_DIR="/old/server/UrbanDreamer/networks/{city}"',
                f'NETWORK_NAME="{city}"',
                f'EDGE_REGISTRY="/old/server/UrbanDreamer/networks/{city}/edge_registry.json"',
                'ROUTE_ARTIFACTS_DIR="/old/server/UrbanDreamer/results/old/records"',
                'WM_CFM_INIT_MODE="zero"',
                'WM_CANDIDATE_EDGE_COST_SOURCE="decoded_speed"',
                'INCIDENT_EDGE="edge#0"',
                'INCIDENT_START="10"',
                'INCIDENT_END="20"',
                'INCIDENT_SPEED="1"',
                'RATIO_MACHINES="0.4"',
                'N_ROUTES="4"',
            ]
        )
        + '\n',
        encoding='utf-8',
    )

    bundle = root / spec.post_bundle_rel
    bundle.mkdir(parents=True, exist_ok=True)
    for name in ('agents.csv', 'routes.csv', 'route.rou.xml'):
        (bundle / name).write_text(name + '\n', encoding='utf-8')
    (bundle / 'post_mutation_hdv_learning_state.json').write_text('{}\n', encoding='utf-8')
    (bundle / 'post_mutation_state.json').write_text(
        json.dumps(
            {
                'schema_version': 1,
                'network': city,
                'ratio_machines': 0.4,
                'should_humans_adapt': False,
                'machine_agent_ids': ['1'],
                'human_agent_ids': ['2'],
                'human_learning_state_ref': '/old/server/hdv.json',
                'artifact_refs': {
                    'agents_csv': '/old/server/agents.csv',
                    'routes_csv': '/old/server/routes.csv',
                    'route_rou_xml': '/old/server/route.rou.xml',
                },
            }
        )
        + '\n',
        encoding='utf-8',
    )

    network = root / 'networks' / city
    network.mkdir(parents=True, exist_ok=True)
    for name in (
        'agents.csv',
        f'od_{city}.txt',
        f'{city}.net.xml',
        'edge_registry.json',
    ):
        (network / name).write_text('{}\n', encoding='utf-8')


def _write_fake_existing_warmup(root: Path, city: str) -> None:
    source = EXISTING_WARMUP_SPECS[city]
    data_dir = root / 'data' / source.run_tag / 'collect_urb_fair_eval_postmutation'
    records_dir = (
        root / 'results' / source.run_tag / 'collect_urb_fair_eval_postmutation_records'
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / f'{source.task_name}.pt').write_bytes(b'fake warmup dataset')

    bundle = root / CITY_SPECS[city].post_bundle_rel

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        'schema_version': 1,
        'network': city,
        'task_name': source.task_name,
        'partial': False,
        'episodes_saved': 50,
        'should_humans_adapt': False,
        'action_schema': 'av_planned_edge_flow_pre_advance_v1',
        'artifact_hashes': {
            'agents_csv': sha256(bundle / 'agents.csv'),
            'routes_csv': sha256(bundle / 'routes.csv'),
            'route_rou_xml': sha256(bundle / 'route.rou.xml'),
        },
    }
    (records_dir / 'urb_fair_collection_manifest.json').write_text(
        json.dumps(manifest) + '\n', encoding='utf-8'
    )


def _default_args(root: Path):
    return build_arg_parser().parse_args(
        [
            '--repo-root',
            str(root),
            '--run-id',
            'test_run',
        ]
    )


def test_three_city_specs_use_network_local_post_mutation_bundles():
    assert CITY_SPECS['gretz_armainvilliers'].post_bundle_rel == (
        'networks/gretz_armainvilliers/post_mutation'
    )
    assert CITY_SPECS['ingolstadt_custom'].post_bundle_rel == (
        'networks/ingolstadt_custom/post_mutation'
    )
    assert CITY_SPECS['fontenay_en_parisis'].post_bundle_rel == (
        'networks/fontenay_en_parisis'
    )
    for spec in CITY_SPECS.values():
        assert not spec.post_bundle_rel.startswith('results/')
    assert CITY_SPECS['fontenay_en_parisis'].sumo_extra_args == (
        '--collision.action warn --ignore-junction-blocker 60 '
        '--tripinfo-output.write-unfinished'
    )
    assert CITY_SPECS['fontenay_en_parisis'].train_sim_buffer_after_last_depart == 1500
    assert CITY_SPECS['gretz_armainvilliers'].train_sim_buffer_after_last_depart is None
    assert CITY_SPECS['ingolstadt_custom'].train_sim_buffer_after_last_depart is None


def test_pipeline_defaults_to_150_episode_sample_efficiency_run(tmp_path):
    args = build_arg_parser().parse_args(['--repo-root', str(tmp_path)])

    assert args.run_id == 'online_wm_mainline_150ep_20260717'
    assert args.online_episodes == 150
    assert args.checkpoint_episodes == '25,50,75,100,125,150'
    assert args.city_workers == 3
    assert args.city_gpus == ''
    assert args.reuse_existing_warmup is False
    assert args.reuse_world_models_from_run_id == ''
    assert args.wm_batch_size == 1
    assert args.wm_grad_accum == 32
    assert args.wm_flow_cache_mb == 512
    assert args.online_wm_batch_size == 4
    assert args.online_wm_space_attention_batch_chunk_size == 16
    assert args.online_imagined_rollout_batch_size == 4
    assert args.train_sim_buffer_after_last_depart == 2000
    assert args.wm_mlp_token_chunk_size == 4096
    assert args.online_wm_mlp_token_chunk_size == 16384
    assert args.wm_time_attention_batch_chunk_size == 256
    assert args.online_wm_time_attention_batch_chunk_size == 32768
    assert args.eval_hdv_variants == ('final', 'original')


def test_fontenay_launcher_uses_dedicated_gpu_wm_profile():
    launcher = (
        ROOT / 'scripts' / 'run_fontenay_online_wm_full_pipeline.sh'
    ).read_text(encoding='utf-8')

    assert '--wm-batch-size 1' in launcher
    assert '--wm-grad-accum 32' in launcher
    assert '--wm-flow-cache-mb 1024' in launcher
    assert '--wm-space-attention-batch-chunk-size 4' in launcher
    assert '--wm-mlp-token-chunk-size 16384' in launcher
    assert '--wm-time-attention-batch-chunk-size 8192' in launcher


def test_eval_hdv_variant_parser_can_select_only_initial_state(tmp_path):
    args = build_arg_parser().parse_args(
        [
            '--repo-root',
            str(tmp_path),
            '--eval-hdv-variants',
            'original',
        ]
    )

    assert args.eval_hdv_variants == ('original',)

    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(
            [
                '--repo-root',
                str(tmp_path),
                '--eval-hdv-variants',
                'unknown',
            ]
        )


def test_existing_warmup_reuse_validates_and_redirects_training_inputs(tmp_path):
    city = 'ingolstadt_custom'
    spec = CITY_SPECS[city]
    _write_fake_city_inputs(tmp_path, city)
    _write_fake_existing_warmup(tmp_path, city)
    output_paths = city_paths(tmp_path, 'test_run', spec)
    prepare_city_inputs(tmp_path, spec, output_paths)

    warmup_paths = resolve_warmup_paths(
        root=tmp_path,
        spec=spec,
        output_paths=output_paths,
        expected_episodes=50,
        reuse_existing=True,
    )

    source = EXISTING_WARMUP_SPECS[city]
    expected_data = tmp_path / 'data' / source.run_tag / 'collect_urb_fair_eval_postmutation'
    expected_records = (
        tmp_path
        / 'results'
        / source.run_tag
        / 'collect_urb_fair_eval_postmutation_records'
    )
    assert warmup_paths.data_dir == expected_data
    assert warmup_paths.records_dir == expected_records
    assert warmup_paths.task_name == source.task_name
    assert warmup_paths.wm_dir == output_paths.wm_dir
    assert warmup_paths.online_dir == output_paths.online_dir

    args = _default_args(tmp_path)
    wm = build_wm_command(
        python_bin='python', root=tmp_path, paths=warmup_paths, args=args
    )
    online = build_online_command(
        python_bin='python',
        root=tmp_path,
        spec=spec,
        paths=warmup_paths,
        checkpoints=(25, 50, 75, 100),
        args=args,
    )
    evaluation = build_eval_command(
        python_bin='python',
        root=tmp_path,
        spec=spec,
        paths=warmup_paths,
        episode=100,
        args=args,
        hdv_variant='original',
    )
    assert _value_after(wm, '--data_dirs') == str(expected_data)
    assert _value_after(wm, '--flow_task_name') == source.task_name
    assert _value_after(online, '--route-artifacts-dir') == str(
        warmup_paths.source_bundle
    )
    assert _value_after(evaluation, '--reuse_artifacts_dir') == str(
        warmup_paths.source_bundle
    )

    provenance = json.loads(output_paths.provenance_json.read_text(encoding='utf-8'))
    assert provenance['protocol']['warmup_reused'] is True
    assert provenance['protocol']['warmup_random_av_fraction'] == 0.6
    assert provenance['protocol']['warmup_epsilon'] == 0.4
    assert provenance['warmup_input']['artifact_hashes_verified'] is True


def test_world_model_reuse_is_per_city_and_falls_back_to_current_run(tmp_path):
    prior_run_id = 'prior_run'
    complete_spec = CITY_SPECS['gretz_armainvilliers']
    incomplete_spec = CITY_SPECS['fontenay_en_parisis']
    complete_source = city_paths(tmp_path, prior_run_id, complete_spec)
    complete_source.wm_dir.mkdir(parents=True)
    for name in ('step_0008000.pt', 'latest.pt', 'flow_projector_latest.pt'):
        (complete_source.wm_dir / name).write_bytes(b'checkpoint')

    complete_output = city_paths(tmp_path, 'new_run', complete_spec)
    incomplete_output = city_paths(tmp_path, 'new_run', incomplete_spec)
    reused = resolve_world_model_paths(
        root=tmp_path,
        spec=complete_spec,
        output_paths=complete_output,
        expected_steps=8000,
        reuse_run_id=prior_run_id,
    )
    fallback = resolve_world_model_paths(
        root=tmp_path,
        spec=incomplete_spec,
        output_paths=incomplete_output,
        expected_steps=8000,
        reuse_run_id=prior_run_id,
    )

    assert reused.wm_dir == complete_source.wm_dir
    assert reused.online_dir == complete_output.online_dir
    assert fallback.wm_dir == incomplete_output.wm_dir


def test_incomplete_world_model_run_resumes_latest_checkpoint(tmp_path):
    spec = CITY_SPECS['fontenay_en_parisis']
    paths = city_paths(tmp_path, 'resume_run', spec)
    paths.wm_dir.mkdir(parents=True)
    latest = paths.wm_dir / 'latest.pt'
    latest.write_bytes(b'checkpoint')

    command = build_wm_command(
        python_bin='python',
        root=tmp_path,
        paths=paths,
        args=_default_args(tmp_path),
    )

    assert _value_after(command, '--resume') == str(latest)


def test_run_cities_starts_all_three_city_pipelines_concurrently(tmp_path, monkeypatch):
    args = _default_args(tmp_path)
    args.dry_run = True
    barrier = threading.Barrier(len(CITY_SPECS))
    started = []
    started_lock = threading.Lock()

    def fake_run_city(*, spec, **kwargs):
        del kwargs
        with started_lock:
            started.append(spec.key)
        barrier.wait(timeout=5)
        return {'city': spec.key}

    monkeypatch.setattr(pipeline, 'run_city', fake_run_city)

    results = run_cities(
        root=tmp_path,
        python_bin=sys.executable,
        cities=tuple(CITY_SPECS.values()),
        checkpoints=(25, 50, 75, 100),
        args=args,
        env={},
        city_workers=3,
    )

    assert set(started) == set(CITY_SPECS)
    assert [result['city'] for result in results] == list(CITY_SPECS)


def test_run_cities_uses_each_gpu_as_an_exclusive_worker_slot(tmp_path, monkeypatch):
    args = _default_args(tmp_path)
    args.dry_run = True
    args.city_gpus = '0,1'
    first_wave = threading.Barrier(2)
    lock = threading.Lock()
    active_by_gpu = {'0': 0, '1': 0}
    max_active_by_gpu = {'0': 0, '1': 0}
    max_active_total = 0
    assignments = []

    def fake_run_city(*, spec, args, env, **kwargs):
        nonlocal max_active_total
        del kwargs
        gpu_id = env['CUDA_VISIBLE_DEVICES']
        assert args.device == 'cuda'
        with lock:
            assignments.append((spec.key, gpu_id))
            active_by_gpu[gpu_id] += 1
            max_active_by_gpu[gpu_id] = max(
                max_active_by_gpu[gpu_id], active_by_gpu[gpu_id]
            )
            max_active_total = max(max_active_total, sum(active_by_gpu.values()))
            is_first_wave = len(assignments) <= 2
        if is_first_wave:
            first_wave.wait(timeout=5)
        with lock:
            active_by_gpu[gpu_id] -= 1
        return {'city': spec.key}

    monkeypatch.setattr(pipeline, 'run_city', fake_run_city)

    results = run_cities(
        root=tmp_path,
        python_bin=sys.executable,
        cities=tuple(CITY_SPECS.values()),
        checkpoints=(25, 50, 75, 100),
        args=args,
        env={},
        city_workers=3,
    )

    assert max_active_total == 2
    assert max_active_by_gpu == {'0': 1, '1': 1}
    assert set(assignments[:2]) == {
        ('gretz_armainvilliers', '0'),
        ('ingolstadt_custom', '1'),
    }
    assert [result['city'] for result in results] == list(CITY_SPECS)


def test_run_cities_allows_other_cities_to_finish_before_reporting_failure(
    tmp_path, monkeypatch
):
    args = _default_args(tmp_path)
    args.dry_run = True
    attempted = []
    attempted_lock = threading.Lock()

    def fake_run_city(*, spec, **kwargs):
        del kwargs
        with attempted_lock:
            attempted.append(spec.key)
        if spec.key == 'gretz_armainvilliers':
            raise RuntimeError('collection reset failed')
        return {'city': spec.key}

    monkeypatch.setattr(pipeline, 'run_city', fake_run_city)

    with pytest.raises(RuntimeError, match='gretz_armainvilliers'):
        run_cities(
            root=tmp_path,
            python_bin=sys.executable,
            cities=tuple(CITY_SPECS.values()),
            checkpoints=(25, 50, 75, 100),
            args=args,
            env={},
            city_workers=3,
        )

    assert set(attempted) == set(CITY_SPECS)


def test_base_environment_replaces_invalid_thread_counts(tmp_path, monkeypatch):
    monkeypatch.setenv('OMP_NUM_THREADS', '')
    monkeypatch.setenv('MKL_NUM_THREADS', 'invalid')

    env = _base_environment(_default_args(tmp_path), tmp_path, sys.executable)

    assert env['OMP_NUM_THREADS'] == '1'
    assert env['MKL_NUM_THREADS'] == '1'


def test_runtime_preflight_reports_missing_routerl_before_collection(tmp_path, monkeypatch):
    for relative in (
        'scripts/collect_urb_fair.py',
        'dreamer4/train_dynamics.py',
        'ablation/run_gretz_online_wm_dreamer_ablation.py',
        'scripts/eval_sumo.py',
        'scripts/select_best_eval_summary.py',
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    sumo_home = tmp_path / 'sumo'
    (sumo_home / 'bin').mkdir(parents=True)
    (sumo_home / 'bin' / 'sumo').touch()
    completed = subprocess.CompletedProcess(
        args=['python', '-c', 'probe'],
        returncode=3,
        stdout='{"errors": {"routerl": "ModuleNotFoundError"}}\n',
        stderr='',
    )
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return completed

    monkeypatch.setattr(subprocess, 'run', fake_run)

    with pytest.raises(RuntimeError, match='routerlurb'):
        runtime_preflight(
            root=tmp_path,
            python_bin='/missing-env/python',
            env={'SUMO_HOME': str(sumo_home)},
            device='cuda',
        )

    command = calls[0][0][0]
    assert command[0] == '/missing-env/python'
    assert 'TrafficEnvironment' in command[2]
    assert 'save_detectors_info' in command[2]
    assert 'importlib.metadata.version("routerlurb")' in command[2]


def test_pipeline_python_default_is_current_interpreter_not_named_conda_env(tmp_path, monkeypatch):
    monkeypatch.delenv('PYTHON_BIN', raising=False)

    args = build_arg_parser().parse_args(['--repo-root', str(tmp_path)])

    assert args.python_bin == sys.executable


def test_prepare_city_inputs_rewrites_settings_and_post_state_for_new_root(tmp_path):
    city = 'gretz_armainvilliers'
    _write_fake_city_inputs(tmp_path, city)
    spec = CITY_SPECS[city]
    paths = city_paths(tmp_path, 'test_run', spec)

    prepare_city_inputs(tmp_path, spec, paths)

    settings_text = paths.portable_settings.read_text(encoding='utf-8')
    assert '/old/server/UrbanDreamer' not in settings_text
    assert f'NETWORK_DIR="{paths.network_dir}"' in settings_text
    assert 'WM_CFM_INIT_MODE="current"' in settings_text
    assert 'WM_CANDIDATE_EDGE_COST_SOURCE="freeflow"' in settings_text

    state = json.loads(paths.portable_post_state.read_text(encoding='utf-8'))
    assert state['human_learning_state_ref'] == str(paths.source_hdv_state.resolve())
    assert state['artifact_refs'] == {
        'agents_csv': str((paths.source_bundle / 'agents.csv').resolve()),
        'routes_csv': str((paths.source_bundle / 'routes.csv').resolve()),
        'route_rou_xml': str((paths.source_bundle / 'route.rou.xml').resolve()),
    }
    provenance = json.loads(paths.provenance_json.read_text(encoding='utf-8'))
    assert provenance['protocol']['warmup_epsilon'] == 0.4
    assert provenance['protocol']['eval_action'] == 'masked_policy_sample'
    assert provenance['protocol']['eval_hdv_variants']['original'] == {
        'state': 'source_post_mutation_state',
        'human_learning_after_mutation': False,
    }
    assert provenance['protocol']['train_sumo_extra_args'] == '--collision.action warn'


def test_pipeline_commands_lock_current_online_wm_protocol(tmp_path):
    city = 'ingolstadt_custom'
    spec = CITY_SPECS[city]
    paths = city_paths(tmp_path, 'test_run', spec)
    args = build_arg_parser().parse_args(
        [
            '--repo-root',
            str(tmp_path),
            '--run-id',
            'test_run',
            '--wm-flow-cache-mb',
            '1024',
        ]
    )
    checkpoints = (25, 50, 75, 100, 125, 150)

    collect = build_collect_command(
        python_bin='python', root=tmp_path, spec=spec, paths=paths, args=args
    )
    assert _value_after(collect, '--random_av_fraction') == '0.0'
    assert _value_after(collect, '--epsilon') == '0.4'
    assert '--disable_post_mutation_hdv_learning' in collect
    assert _value_after(collect, '--post_mutation_state_in') == str(paths.portable_post_state)
    assert _value_after(collect, '--hdv_state_in') == str(paths.source_hdv_state)

    wm = build_wm_command(
        python_bin='python', root=tmp_path, paths=paths, args=args
    )
    assert _value_after(wm, '--transition_cfm_init_mode') == 'current'
    assert _value_after(wm, '--adm_cfm_init_mode') == 'current'
    assert _value_after(wm, '--flow_cache_mb') == '1024'
    assert '--gradient_checkpointing' in wm
    assert '--edge_action_injection' in wm
    assert '--graph_cond_edge_conditioning' in wm

    online = build_online_command(
        python_bin='python',
        root=tmp_path,
        spec=spec,
        paths=paths,
        checkpoints=checkpoints,
        args=args,
    )
    assert _value_after(online, '--real-policy-route-selection') == 'sample'
    assert _value_after(online, '--eval-policy-route-selection') == 'sample'
    assert _value_after(online, '--train-episodes') == '150'
    assert _value_after(online, '--checkpoint-episodes') == '25,50,75,100,125,150'
    assert _value_after(online, '--sim-buffer-after-last-depart') == '2000'
    assert _value_after(online, '--global-advantage-weight') == '1.0'
    assert _value_after(online, '--imagined-update-passes-per-collect') == '16'
    assert _value_after(online, '--imagined-rollouts-per-update') == '8'
    assert _value_after(online, '--imagined-rollout-batch-size') == '4'
    assert _value_after(online, '--wm-hdv-target-source') == 'static_edge_demand'
    assert _value_after(online, '--wm-batch-size') == '4'
    assert '--wm-gradient-checkpointing' in online
    assert _value_after(online, '--wm-space-attention-batch-chunk-size') == '16'
    assert _value_after(online, '--wm-mlp-token-chunk-size') == '16384'
    assert _value_after(online, '--wm-time-attention-batch-chunk-size') == '32768'
    assert _value_after(online, '--imagined-hdv-context-source') == 'causal_response_predictor'
    assert '--enable-hdv-response-predictor' in online
    assert '--keep-human-learning-after-mutation' in online
    assert '--skip-eval' in online


def test_eval_command_uses_final_hdv_state_and_stochastic_policy(tmp_path):
    spec = CITY_SPECS['fontenay_en_parisis']
    paths = city_paths(tmp_path, 'test_run', spec)
    args = _default_args(tmp_path)

    command = build_eval_command(
        python_bin='python',
        root=tmp_path,
        spec=spec,
        paths=paths,
        episode=100,
        args=args,
    )

    assert _value_after(command, '--av_policy_route_selection') == 'sample'
    assert _value_after(command, '--test_eps') == '5'
    assert _value_after(command, '--hdv_state_in') == str(
        paths.online_dir / 'hdv_learning_state_after_online_wm_train.json'
    )
    assert _value_after(command, '--wm_cfm_init_mode') == 'current'
    assert '--deterministic' in command
    assert '--keep_human_learning_after_mutation' in command
    assert _value_after(command, '--sumo_extra_args') == (
        '--collision.action warn --ignore-junction-blocker 60 '
        '--tripinfo-output.write-unfinished'
    )
    assert 'checkpoint_ep00100.pt' in _value_after(command, '--policy_ckpt')
    assert 'wm_projector_ep00100.pt' in _value_after(command, '--projector_ckpt')


def test_original_hdv_eval_uses_source_state_and_freezes_learning(tmp_path):
    spec = CITY_SPECS['fontenay_en_parisis']
    paths = city_paths(tmp_path, 'test_run', spec)
    args = _default_args(tmp_path)

    command = build_eval_command(
        python_bin='python',
        root=tmp_path,
        spec=spec,
        paths=paths,
        episode=100,
        args=args,
        hdv_variant='original',
    )

    assert _value_after(command, '--hdv_state_in') == str(paths.source_hdv_state)
    assert _value_after(command, '--av_policy_route_selection') == 'sample'
    assert '--disable_human_learning_after_mutation' in command
    assert '--keep_human_learning_after_mutation' not in command
    assert _value_after(command, '--results_dir') == str(
        paths.online_dir / 'eval_original_hdv' / 'checkpoint_ep00100'
    )
    assert _value_after(command, '--sumo_extra_args') == (
        '--collision.action warn --ignore-junction-blocker 60 '
        '--tripinfo-output.write-unfinished'
    )


def test_run_city_can_evaluate_only_the_initial_hdv_variant(tmp_path, monkeypatch):
    city = 'fontenay_en_parisis'
    spec = CITY_SPECS[city]
    _write_fake_city_inputs(tmp_path, city)
    _write_fake_existing_warmup(tmp_path, city)
    args = _default_args(tmp_path)
    args.reuse_existing_warmup = True
    args.start_stage = 'eval'
    args.eval_hdv_variants = ('original',)
    checkpoints = (100,)
    manifest_variants = []
    commands = []

    monkeypatch.setattr(pipeline, '_online_complete', lambda *args, **kwargs: True)
    monkeypatch.setattr(pipeline, '_eval_complete', lambda *args, **kwargs: True)
    monkeypatch.setattr(
        pipeline,
        '_write_eval_manifest',
        lambda paths, episodes, variant: manifest_variants.append(variant),
    )
    monkeypatch.setattr(
        pipeline,
        'run_command',
        lambda command, **kwargs: commands.append(command),
    )

    result = run_city(
        root=tmp_path,
        python_bin='python',
        spec=spec,
        checkpoints=checkpoints,
        args=args,
        env={},
    )

    assert manifest_variants == ['original']
    assert len(commands) == 1
    assert _value_after(commands[0], '--output').endswith(
        'best_checkpoint_selection_original_hdv.json'
    )
    assert _value_after(commands[0], '--metric') == 't_all_mean'
    status = json.loads(Path(result['status']).read_text(encoding='utf-8'))
    assert set(status['best_checkpoint_selections']) == {'original'}
    provenance = json.loads(
        city_paths(tmp_path, 'test_run', spec).provenance_json.read_text(encoding='utf-8')
    )
    assert set(provenance['protocol']['eval_hdv_variants']) == {'original'}
    assert result['eval_hdv_variants'] == ['original']
    assert 'eval_summary' not in result
    assert 'best_checkpoint_selection' not in result
    assert result['eval_original_hdv_summary'].endswith(
        'online_wm_eval_original_hdv_summary.json'
    )


def test_fontenay_blocker60_is_used_throughout_pipeline(tmp_path):
    spec = CITY_SPECS['fontenay_en_parisis']
    paths = city_paths(tmp_path, 'test_run', spec)
    args = _default_args(tmp_path)

    collect = build_collect_command(
        python_bin='python', root=tmp_path, spec=spec, paths=paths, args=args
    )
    online = build_online_command(
        python_bin='python',
        root=tmp_path,
        spec=spec,
        paths=paths,
        checkpoints=(25, 50, 75, 100),
        args=args,
    )

    expected = (
        '--collision.action warn --ignore-junction-blocker 60 '
        '--tripinfo-output.write-unfinished'
    )
    assert _value_after(collect, '--sumo_extra_args') == expected
    assert _value_after(online, '--sumo-extra-args') == expected
    assert _value_after(online, '--sim-buffer-after-last-depart') == '1500'
