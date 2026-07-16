import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_three_city_online_wm_full_pipeline import (  # noqa: E402
    CITY_SPECS,
    build_arg_parser,
    build_collect_command,
    build_eval_command,
    build_online_command,
    build_wm_command,
    city_paths,
    prepare_city_inputs,
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


def _default_args(root: Path):
    return build_arg_parser().parse_args(
        [
            '--repo-root',
            str(root),
            '--run-id',
            'test_run',
        ]
    )


def test_three_city_specs_use_expected_post_mutation_bundles():
    assert '20260530_odcompact' in CITY_SPECS['gretz_armainvilliers'].post_bundle_rel
    assert 'current_to_dest_incident_20260528' in CITY_SPECS['ingolstadt_custom'].post_bundle_rel
    assert 'both494_incident_warmup50_fastcap_20260609' in (
        CITY_SPECS['fontenay_en_parisis'].post_bundle_rel
    )
    assert CITY_SPECS['fontenay_en_parisis'].sumo_extra_args == (
        '--collision.action warn --ignore-junction-blocker 60 '
        '--tripinfo-output.write-unfinished'
    )


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
    args = _default_args(tmp_path)
    checkpoints = (50, 100, 200, 400)

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
    assert _value_after(online, '--global-advantage-weight') == '1.0'
    assert _value_after(online, '--imagined-update-passes-per-collect') == '16'
    assert _value_after(online, '--imagined-rollouts-per-update') == '8'
    assert _value_after(online, '--wm-hdv-target-source') == 'static_edge_demand'
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
        episode=400,
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
    assert 'checkpoint_ep00400.pt' in _value_after(command, '--policy_ckpt')
    assert 'wm_projector_ep00400.pt' in _value_after(command, '--projector_ckpt')


def test_original_hdv_eval_uses_source_state_and_freezes_learning(tmp_path):
    spec = CITY_SPECS['fontenay_en_parisis']
    paths = city_paths(tmp_path, 'test_run', spec)
    args = _default_args(tmp_path)

    command = build_eval_command(
        python_bin='python',
        root=tmp_path,
        spec=spec,
        paths=paths,
        episode=400,
        args=args,
        hdv_variant='original',
    )

    assert _value_after(command, '--hdv_state_in') == str(paths.source_hdv_state)
    assert _value_after(command, '--av_policy_route_selection') == 'sample'
    assert '--disable_human_learning_after_mutation' in command
    assert '--keep_human_learning_after_mutation' not in command
    assert _value_after(command, '--results_dir') == str(
        paths.online_dir / 'eval_original_hdv' / 'checkpoint_ep00400'
    )
    assert _value_after(command, '--sumo_extra_args') == (
        '--collision.action warn --ignore-junction-blocker 60 '
        '--tripinfo-output.write-unfinished'
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
        checkpoints=(50, 100, 200, 400),
        args=args,
    )

    expected = (
        '--collision.action warn --ignore-junction-blocker 60 '
        '--tripinfo-output.write-unfinished'
    )
    assert _value_after(collect, '--sumo_extra_args') == expected
    assert _value_after(online, '--sumo-extra-args') == expected
