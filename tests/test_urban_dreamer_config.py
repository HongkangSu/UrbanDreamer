from __future__ import annotations

from pathlib import Path

import pytest

from urban_dreamer.workflow.config import CodecConfig, ConfigError, load_config


def _write_config(path: Path, *, extra: str = "") -> Path:
    path.write_text(
        """
[inputs]
network_name = "toy_network"
network_dir = "network"

[run]
output_dir = "output"
device = "cpu"
seed = 42
warmup_episodes = 2
world_model_initialization_epochs = 3
online_training_episodes = 4
evaluation_episodes = 5

[simulation]
vehicle_max_speed_mps = 40.0
world_model_intervals_per_episode = 60
warmup_horizon_seconds = 3600
warmup_completion_drain_seconds = 7200
warmup_completion_patience_seconds = 600
episode_wall_timeout_seconds = 3600.0
training_completion_buffer_seconds = 2000
evaluation_horizon_seconds = 9000
incident_edge = "edge-1"
incident_start_seconds = 500.0
incident_end_seconds = 2500.0
incident_speed_mps = 0.1
sumo_extra_args = ["--collision.action", "warn"]

[resources]
world_model_initialization_batch_size = 1
world_model_initialization_sequence_length = 16
world_model_gradient_accumulation_steps = 32
online_world_model_batch_size = 4
"""
        + extra,
        encoding="utf-8",
    )
    return path


def test_load_config_resolves_paths_relative_to_toml(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path / "mainline.toml"))

    assert config.inputs.network_name == "toy_network"
    assert config.inputs.network_dir == (tmp_path / "network").resolve()
    assert config.run.output_dir == (tmp_path / "output").resolve()
    assert config.run.device == "cpu"
    assert config.run.online_training_episodes == 4
    assert config.simulation.sumo_extra_args == (
        "--collision.action",
        "warn",
    )
    assert config.codec == CodecConfig()
    assert config.run.codec_pretraining_epochs == 10
    assert config.method.transition_volume_scope == 'previous_state_excluding_current_births'


def test_semantic_fingerprint_ignores_paths_output_and_device(tmp_path: Path) -> None:
    first_path = _write_config(tmp_path / "first.toml")
    second_dir = tmp_path / "relocated"
    second_dir.mkdir()
    second_path = _write_config(second_dir / "second.toml")
    second_path.write_text(
        second_path.read_text(encoding="utf-8")
        .replace('output_dir = "output"', 'output_dir = "another-output"')
        .replace('device = "cpu"', 'device = "cuda"'),
        encoding="utf-8",
    )

    first = load_config(first_path)
    second = load_config(second_path)

    assert first.semantic_fingerprint == second.semantic_fingerprint


def test_semantic_fingerprint_changes_with_training_semantics(tmp_path: Path) -> None:
    first = load_config(_write_config(tmp_path / "first.toml"))
    second_path = _write_config(tmp_path / "second.toml")
    second_path.write_text(
        second_path.read_text(encoding="utf-8").replace(
            "online_training_episodes = 4",
            "online_training_episodes = 5",
        ),
        encoding="utf-8",
    )

    second = load_config(second_path)

    assert first.semantic_fingerprint != second.semantic_fingerprint


def test_load_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = _write_config(tmp_path / "mainline.toml", extra="\n[experiment]\nmode = 'ablation'\n")

    with pytest.raises(ConfigError, match="unknown top-level section"):
        load_config(path)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("warmup_episodes = 2", "warmup_episodes = -1", "warmup_episodes"),
        (
            "incident_end_seconds = 2500.0",
            "incident_end_seconds = 400.0",
            "incident_end_seconds",
        ),
        (
            "world_model_initialization_batch_size = 1",
            "world_model_initialization_batch_size = 0",
            "world_model_initialization_batch_size",
        ),
    ],
)
def test_load_config_rejects_invalid_ranges(
    tmp_path: Path,
    field: str,
    replacement: str,
    message: str,
) -> None:
    path = _write_config(tmp_path / "mainline.toml")
    path.write_text(
        path.read_text(encoding="utf-8").replace(field, replacement),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match=message):
        load_config(path)


@pytest.mark.parametrize("argument", ["--step-length=2", "--step-length", "-c", "--configuration-file=x", "--route-files=x", "--net-file=x", "--begin=1", "--default.speeddev=1", "--default.speedfactor=2", "--additional-files=x"])
def test_sumo_args_cannot_override_safety_contract(tmp_path: Path, argument: str) -> None:
    path = _write_config(tmp_path / "mainline.toml")
    path.write_text(path.read_text().replace('["--collision.action", "warn"]', f'["{argument}"]'))
    with pytest.raises(ConfigError, match="reserved SUMO"):
        load_config(path)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_speed_cap_must_be_finite_positive(tmp_path: Path, value: str) -> None:
    path = _write_config(tmp_path / "mainline.toml")
    path.write_text(path.read_text().replace("vehicle_max_speed_mps = 40.0", f"vehicle_max_speed_mps = {value}"))
    with pytest.raises(ConfigError, match="vehicle_max_speed_mps"):
        load_config(path)


def test_rejects_old_interval_key(tmp_path: Path) -> None:
    path = _write_config(tmp_path / 'mainline.toml')
    path.write_text(path.read_text().replace('world_model_intervals_per_episode', 'decision_intervals_per_episode'))
    with pytest.raises(ConfigError, match='unknown.*decision_intervals_per_episode'):
        load_config(path)


def test_one_transition_sequence_is_valid(tmp_path: Path) -> None:
    path = _write_config(tmp_path / 'mainline.toml')
    path.write_text(path.read_text().replace('world_model_initialization_sequence_length = 16', 'world_model_initialization_sequence_length = 1'))
    assert load_config(path).resources.world_model_initialization_sequence_length == 1


def test_incident_cannot_exceed_vehicle_speed_cap(tmp_path: Path) -> None:
    path = _write_config(tmp_path / 'mainline.toml')
    path.write_text(path.read_text().replace('incident_speed_mps = 0.1', 'incident_speed_mps = 41.0'))
    with pytest.raises(ConfigError, match='incident_speed_mps.*vehicle_max_speed_mps'):
        load_config(path)


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf'])
def test_poisson_demand_duration_must_be_positive_and_finite(tmp_path: Path, value: str) -> None:
    path = _write_config(tmp_path / 'mainline.toml')
    path.write_text(path.read_text().replace('[simulation]', f'[simulation]\ndemand_duration_seconds = {value}'))
    with pytest.raises(ConfigError, match='demand_duration_seconds'):
        load_config(path)


def test_demand_duration_changes_semantics_without_extending_simulation(tmp_path: Path) -> None:
    path = _write_config(tmp_path / 'mainline.toml')
    baseline = load_config(path)
    path.write_text(path.read_text().replace('[simulation]', '[simulation]\ndemand_duration_seconds = 420.0'))
    poisson = load_config(path)
    assert poisson.simulation.demand_duration_seconds == 420.
    assert baseline.simulation.world_model_intervals_per_episode == poisson.simulation.world_model_intervals_per_episode
    assert baseline.semantic_fingerprint != poisson.semantic_fingerprint


def test_legacy_step_budget_is_rejected(tmp_path):
    path = _write_config(tmp_path / "legacy.toml")
    path.write_text(path.read_text().replace('world_model_initialization_epochs',
                                            'world_model_initialization_steps'))
    with pytest.raises(ConfigError, match='unknown.*world_model_initialization_steps'):
        load_config(path)


def test_deprecated_warmup_episode_budget_is_optional_and_nonnegative(tmp_path):
    path = _write_config(tmp_path / 'mainline.toml')
    path.write_text(path.read_text().replace('warmup_episodes = 2\n', ''))
    config = load_config(path)
    assert config.run.warmup_episodes == 0
    assert config.resolved_snapshot()['run']['warmup_episodes'] == 0
    path.write_text(path.read_text().replace('[run]', '[run]\nwarmup_episodes = 0'))
    assert load_config(path).semantic_fingerprint == config.semantic_fingerprint


def test_legacy_collector_warmup_fields_parse_without_changing_mainline_semantics(tmp_path):
    first = _write_config(tmp_path / 'first.toml')
    second = _write_config(tmp_path / 'second.toml')
    second.write_text(second.read_text()
        .replace('warmup_episodes = 2', 'warmup_episodes = 250')
        .replace('warmup_horizon_seconds = 3600', 'warmup_horizon_seconds = 10000')
        .replace('warmup_completion_drain_seconds = 7200', 'warmup_completion_drain_seconds = 600')
        .replace('warmup_completion_patience_seconds = 600', 'warmup_completion_patience_seconds = 0'))
    baseline, legacy = load_config(first), load_config(second)
    assert legacy.run.warmup_episodes == 250
    assert legacy.simulation.warmup_horizon_seconds == 10000
    assert legacy.resolved_snapshot()['simulation']['warmup_completion_drain_seconds'] == 600
    assert baseline.semantic_fingerprint == legacy.semantic_fingerprint


def test_deprecated_simulation_warmup_fields_are_optional_with_collector_defaults(tmp_path):
    path = _write_config(tmp_path / 'mainline.toml')
    lines = [line for line in path.read_text().splitlines() if not line.startswith('warmup_')]
    path.write_text('\n'.join(lines) + '\n')
    config = load_config(path)
    assert config.run.warmup_episodes == 0
    assert config.simulation.warmup_horizon_seconds == 1800
    assert config.simulation.warmup_completion_drain_seconds == 3000
    assert config.simulation.warmup_completion_patience_seconds == 0
    snapshot = config.resolved_snapshot()
    assert snapshot['simulation']['warmup_horizon_seconds'] == 1800
    assert snapshot['simulation']['warmup_completion_drain_seconds'] == 3000
    assert snapshot['simulation']['warmup_completion_patience_seconds'] == 0


def test_shipped_configs_use_offline_pretraining_contract():
    import tomllib
    configs = Path(__file__).parents[1] / 'configs'
    for path in configs.glob('*.toml'):
        raw = tomllib.loads(path.read_text())
        config = load_config(path)
        assert 'warmup_episodes' not in raw['run']
        assert not any(key.startswith('warmup_') for key in raw['simulation'])
        assert config.run.warmup_episodes == 0
        if path.name == 'ingolstadt_actor_warmstart.toml':
            # This isolated profile copies the completed Ingo run's cache and
            # execution budget, rather than the generic archive-pretrain preset.
            assert config.method.sp_prior_weight == 1.
            assert config.data.pretrain_cache_manifest is not None
            assert config.resources.imagination_passes_per_episode == 4
            assert config.run.online_training_episodes == 2000
            continue
        assert config.run.output_dir.name.endswith('_pretrain_eventq_meso')
        assert config.simulation.traffic_model == 'meso'
        assert config.simulation.backend == 'libsumo'
        assert config.simulation.meso_junction_control
        assert not config.simulation.meso_lane_queue
        assert not config.simulation.meso_multi_queue
        assert config.simulation.meso_interpolate_positions
        assert config.run.world_model_initialization_epochs > 0
        assert config.run.codec_pretraining_epochs == 10
        assert config.codec.speed_positive
        assert config.method.transition_volume_scope == 'previous_state_excluding_current_births'
        assert config.resources.world_model_initialization_sequence_length == 1
        assert config.method.fm_training_mode == "single_step"
        assert config.resources.world_model_pretrain_space_chunk == 16
        assert config.resources.world_model_pretrain_time_chunk == 32768
        assert config.resources.world_model_pretrain_mlp_chunk == 16384
        assert config.resources.world_model_pretrain_activation_checkpointing
        assert 'codec' in raw
        assert 'data' in raw
        if config.inputs.network_name in {'ingolstadt_custom', 'gretz_armainvilliers', 'fontenay_en_parisis'}:
            assert len(config.data.archive_roots) == 1
            assert config.data.archive_roots[0].name == 'city_baseline_dataset_meso'
            assert {root.name for root in config.data.archive_roots} == {'city_baseline_dataset_meso'}
        elif config.inputs.network_name == 'toy_cross_bottleneck':
            assert config.data.archive_roots == ()


def test_codec_configuration_and_epoch_budget_change_checkpoint_semantics(tmp_path):
    path = _write_config(tmp_path / 'mainline.toml')
    baseline = load_config(path)
    path.write_text(path.read_text() + '\n[codec]\nbatch_size=8\nmax_updates=50\n'
                    'validation_every=7\noccupied_vehicle_mae_max=0.25\nspeed_positive=false\n')
    configured = load_config(path)
    assert configured.codec.batch_size == 8 and configured.codec.max_updates == 50
    assert configured.codec.validation_every == 7
    assert configured.codec.occupied_vehicle_mae_max == .25
    assert not configured.codec.speed_positive
    assert configured.semantic_fingerprint != baseline.semantic_fingerprint
    assert configured.resolved_snapshot()['codec']['learning_rate'] == 1e-4
    path.write_text(path.read_text().replace('[run]', '[run]\ncodec_pretraining_epochs=3'))
    assert load_config(path).semantic_fingerprint != configured.semantic_fingerprint


@pytest.mark.parametrize('setting', [
    'learning_rate=0', 'learning_rate=nan', 'batch_size=0', 'batch_size=true',
    'max_updates=-1', 'validation_every=0', 'occupied_vehicle_mae_max=-1',
    'vehicle_mae_max=inf', 'empty_vehicle_mae_max=-0.01',
    'total_relative_mae_max=nan', 'speed_mae_mps_max=-1', 'speed_positive=1'])
def test_codec_config_rejects_invalid_budgets_and_quality_thresholds(tmp_path, setting):
    path = _write_config(tmp_path / 'mainline.toml', extra='\n[codec]\n' + setting)
    with pytest.raises(ConfigError, match=r'\[codec\]'):
        load_config(path)


@pytest.mark.parametrize('value', ['0', '-1', 'true'])
def test_codec_epoch_upper_bound_is_positive_integer(tmp_path, value):
    path = _write_config(tmp_path / 'mainline.toml')
    path.write_text(path.read_text().replace('[run]', '[run]\ncodec_pretraining_epochs=' + value))
    with pytest.raises(ConfigError, match='codec_pretraining_epochs'):
        load_config(path)


@pytest.mark.parametrize("legacy_scope", ["all_vehicles", "initial_cohort"])
def test_per_transition_volume_scope_is_required_for_new_mainline(tmp_path, legacy_scope):
    path = _write_config(tmp_path / 'mainline.toml',
                         extra=f'\n[method]\ntransition_volume_scope="{legacy_scope}"\n')
    with pytest.raises(ConfigError, match='transition_volume_scope must be previous_state_excluding_current_births'):
        load_config(path)


def test_unknown_codec_fields_fail_closed(tmp_path):
    path = _write_config(tmp_path / 'mainline.toml', extra='\n[codec]\nmae_threshold=1\n')
    with pytest.raises(ConfigError, match='unknown.*codec.*mae_threshold'):
        load_config(path)


def test_optional_cache_manifests_and_explicit_validation_roots_resolve_from_config(tmp_path):
    path = _write_config(tmp_path / 'mainline.toml', extra='''
[data]
archive_roots = ["training"]
validation_archive_roots = ["heldout"]
pretrain_cache_manifest = "caches/training/manifest.json"
validation_cache_manifest = "caches/validation/manifest.json"
''')
    config = load_config(path)
    assert config.data.pretrain_cache_manifest == tmp_path / 'caches/training/manifest.json'
    assert config.data.validation_cache_manifest == tmp_path / 'caches/validation/manifest.json'
    assert config.data.validation_archive_roots == (tmp_path / 'heldout',)
    snapshot = config.resolved_snapshot()['data']
    assert snapshot['pretrain_cache_manifest'] == str(config.data.pretrain_cache_manifest)
    defaults = load_config(_write_config(tmp_path / 'defaults.toml'))
    assert defaults.data.pretrain_cache_manifest is defaults.data.validation_cache_manifest is None
    assert defaults.data.validation_archive_roots == ()
    assert defaults.resolved_snapshot()['data']['pretrain_cache_manifest'] is None


def test_cache_paths_are_bound_by_runtime_identity_without_changing_semantic_fingerprint(tmp_path):
    path = _write_config(tmp_path / 'first.toml', extra='''
[data]
pretrain_cache_manifest = "first/training.json"
validation_cache_manifest = "first/validation.json"
validation_archive_roots = ["first/heldout"]
''')
    first = load_config(path)
    relocated = tmp_path / 'relocated.toml'
    relocated.write_text(path.read_text().replace('first/', 'second/'))
    second = load_config(relocated)
    assert first.data.pretrain_cache_manifest != second.data.pretrain_cache_manifest
    assert first.semantic_fingerprint == second.semantic_fingerprint
    raw_sources = load_config(_write_config(tmp_path / 'raw.toml', extra='\n[data]\narchive_roots=["first/training"]\n'))
    assert first.semantic_fingerprint != raw_sources.semantic_fingerprint


@pytest.mark.parametrize('setting', [
    'pretrain_cache_manifest=""', 'validation_cache_manifest=" "',
    'pretrain_cache_manifest=1', 'validation_cache_manifest=true',
    'validation_archive_roots="heldout"', 'validation_archive_roots=[""]',
    'validation_archive_roots=[1]'])
def test_invalid_cache_paths_and_validation_roots_are_rejected(tmp_path, setting):
    path = _write_config(tmp_path / 'mainline.toml', extra='\n[data]\n' + setting)
    with pytest.raises(ConfigError, match=r'\[data\]'):
        load_config(path)


def test_pretraining_runtime_resources_have_conservative_defaults_and_fingerprinted_overrides(tmp_path):
    path = _write_config(tmp_path / 'mainline.toml')
    baseline = load_config(path)
    assert baseline.resources.world_model_pretrain_space_chunk == 1
    assert baseline.resources.world_model_pretrain_time_chunk == 256
    assert baseline.resources.world_model_pretrain_mlp_chunk == 4096
    assert baseline.resources.world_model_pretrain_activation_checkpointing
    path.write_text(path.read_text().replace('[resources]', '''[resources]
world_model_pretrain_space_chunk = 16
world_model_pretrain_time_chunk = 32768
world_model_pretrain_mlp_chunk = 16384
world_model_pretrain_activation_checkpointing = false'''))
    faster = load_config(path)
    assert faster.resources.world_model_pretrain_space_chunk == 16
    assert faster.resources.world_model_pretrain_time_chunk == 32768
    assert faster.resources.world_model_pretrain_mlp_chunk == 16384
    assert not faster.resources.world_model_pretrain_activation_checkpointing
    assert faster.semantic_fingerprint != baseline.semantic_fingerprint
    assert faster.resolved_snapshot()['resources']['world_model_pretrain_space_chunk'] == 16


@pytest.mark.parametrize('setting', [
    'world_model_pretrain_space_chunk=0', 'world_model_pretrain_time_chunk=-1',
    'world_model_pretrain_mlp_chunk=true', 'world_model_pretrain_space_chunk=1.5',
    'world_model_pretrain_activation_checkpointing=1',
    'world_model_pretrain_activation_checkpointing="off"'])
def test_pretrain_resource_profile_knobs_are_strictly_typed(tmp_path, setting):
    path = _write_config(tmp_path / 'mainline.toml')
    path.write_text(path.read_text().replace('[resources]', '[resources]\n' + setting))
    with pytest.raises(ConfigError, match=r'\[resources\]'):
        load_config(path)


def test_precision_and_exposure_readout_have_explicit_fingerprinted_contracts(tmp_path):
    path = _write_config(tmp_path / 'config.toml')
    first = load_config(path)
    assert first.resources.world_model_pretrain_precision == 'fp32'
    assert first.method.exposure_readout_supervision == 'observed_endpoint'
    path.write_text(path.read_text().replace('[resources]', '[resources]\nworld_model_pretrain_precision="bf16"')
                    + '\n[method]\nexposure_readout_supervision="sampled_endpoint"\n')
    second = load_config(path)
    assert second.resources.world_model_pretrain_precision == 'bf16'
    assert second.method.exposure_readout_supervision == 'sampled_endpoint'
    assert second.semantic_fingerprint != first.semantic_fingerprint


@pytest.mark.parametrize('extra,message', [
    ('\n[method]\nexposure_readout_supervision="guessed"\n', 'exposure_readout_supervision'),
    ('\n[method]\nexposure_readout_supervision=true\n', 'exposure_readout_supervision'),
])
def test_invalid_exposure_supervision_is_rejected(tmp_path, extra, message):
    path = _write_config(tmp_path / 'config.toml', extra=extra)
    with pytest.raises(ConfigError, match=message):
        load_config(path)


@pytest.mark.parametrize('value', ['"fp16"', '"float32"', 'true', '1'])
def test_pretraining_precision_is_only_fp32_or_bf16(tmp_path, value):
    path = _write_config(tmp_path / 'config.toml')
    path.write_text(path.read_text().replace('[resources]', '[resources]\nworld_model_pretrain_precision=' + value))
    with pytest.raises(ConfigError, match='world_model_pretrain_precision'):
        load_config(path)



def test_omitted_fm_solver_retains_legacy_defaults_and_fingerprint(tmp_path):
    omitted = load_config(_write_config(tmp_path / 'omitted.toml'))
    explicit = load_config(_write_config(tmp_path / 'explicit.toml',
        extra='\n[method]\nfm_steps = 2\nfm_solver = "euler"\n'))
    assert omitted.method.fm_steps == 2
    assert omitted.method.fm_solver == 'euler'
    assert omitted.semantic_fingerprint == explicit.semantic_fingerprint


def test_midpoint_solver_is_accepted_and_fingerprinted(tmp_path):
    euler = load_config(_write_config(tmp_path / 'euler.toml'))
    midpoint = load_config(_write_config(tmp_path / 'midpoint.toml',
        extra='\n[method]\nfm_steps = 100\nfm_solver = "midpoint"\n'))
    assert midpoint.method.fm_steps == 100
    assert midpoint.method.fm_solver == 'midpoint'
    assert midpoint.semantic_fingerprint != euler.semantic_fingerprint
    same_steps_euler = load_config(_write_config(tmp_path / 'steps100.toml',
        extra='\n[method]\nfm_steps = 100\nfm_solver = "euler"\n'))
    assert midpoint.semantic_fingerprint != same_steps_euler.semantic_fingerprint


def test_all_presets_select_euler_100_without_recursive_training():
    paths = list((Path(__file__).resolve().parents[1] / 'configs').glob('*.toml'))
    assert paths
    for path in paths:
        config = load_config(path)
        assert config.method.fm_steps == 100, path
        assert config.method.fm_solver == 'euler', path
        assert config.method.fm_training_mode == 'single_step', path
        assert config.resources.world_model_initialization_sequence_length == 1, path



def test_presets_use_verified_imagination_compile_profiles():
    paths = list((Path(__file__).resolve().parents[1] / 'configs').glob('*.toml'))
    assert paths
    for path in paths:
        config = load_config(path)
        resource = config.resources
        assert resource.world_model_imagination_precision == ('bf16' if path.name in {
            'ingolstadt_custom.toml', 'ingolstadt_custom_offline.toml',
            'ingolstadt_actor_warmstart.toml'} else 'fp32'), path
        assert resource.world_model_imagination_compile is True, path
        assert resource.world_model_imagination_compile_mode == 'default', path
        assert resource.world_model_imagination_space_chunk == 16, path
        assert resource.world_model_imagination_time_chunk == 32768, path
        assert resource.world_model_imagination_mlp_chunk == 16384, path
        assert config.method.fm_solver == 'euler' and config.method.fm_steps == 100, path
        assert config.method.disagreement_samples == 2, path


def test_omitted_imagination_compile_retains_legacy_disabled_default(tmp_path):
    config = load_config(_write_config(tmp_path / 'legacy.toml'))
    assert config.resources.world_model_imagination_compile is False
    assert config.resources.world_model_imagination_precision == 'fp32'
