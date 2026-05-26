import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_urb_fair_full_cycle import (  # noqa: E402
    build_plan,
    derive_agent_counts,
    parse_args,
    parse_checkpoint_episodes,
    write_plan_settings,
)


def _write_network(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "agents.csv").write_text(
        "id,start_time,origin,destination,kind\n"
        "0,0,0,0,Human\n"
        "1,0,0,1,Human\n"
        "2,0,1,0,Human\n"
        "3,0,1,1,Human\n"
        "4,0,2,0,Human\n",
        encoding="utf-8",
    )
    (path / "routes.csv").write_text("origins,destinations,path,free_flow_time\n0,0,a b,1.0\n", encoding="utf-8")
    (path / "route.rou.xml").write_text("<routes></routes>\n", encoding="utf-8")
    (path / "edge_registry.json").write_text(json.dumps({"edges": []}), encoding="utf-8")
    (path / "route_registry.json").write_text(json.dumps({}), encoding="utf-8")
    (path / "od_toy_net.txt").write_text("0 1 1\n", encoding="utf-8")


def test_parse_checkpoint_episodes_accepts_space_or_comma_lists():
    assert parse_checkpoint_episodes("100 200,300") == [100, 200, 300]


def test_parse_args_accepts_resume_start_stage(tmp_path):
    args = parse_args(
        [
            "--network-dir",
            str(tmp_path),
            "--network-name",
            "toy_net",
            "--start-stage",
            "train_dynamics",
        ]
    )

    assert args.start_stage == "train_dynamics"


def test_derive_agent_counts_uses_agents_csv_and_ratio(tmp_path):
    network = tmp_path / "network"
    _write_network(network)

    assert derive_agent_counts(network / "agents.csv", ratio_machines=0.4) == (2, 3)


def test_full_cycle_runner_translates_public_network_flags(tmp_path):
    network = tmp_path / "network"
    _write_network(network)
    state_in = tmp_path / "existing_post_mutation_state.json"
    state_in.write_text("{}", encoding="utf-8")

    args = parse_args(
        [
            "--network-dir",
            str(network),
            "--network-name",
            "toy_net",
            "--generated-config-dir",
            str(tmp_path / "generated"),
            "--output-root",
            str(tmp_path / "data"),
            "--logs-root",
            str(tmp_path / "logs"),
            "--results-root",
            str(tmp_path / "results"),
            "--checkpoint-episodes",
            "100 200",
            "--incident-edge",
            "201201950#0",
            "--incident-start",
            "500",
            "--incident-end",
            "1500",
            "--incident-speed",
            "1.0",
            "--python-bin",
            "python",
            "--collect-post-mutation-state-in",
            str(state_in),
        ]
    )
    plan = build_plan(args)
    write_plan_settings(args, plan)

    assert plan.collect_command[plan.collect_command.index("--settings") + 1].endswith("urb_fair_collect.env")
    assert plan.collect_command[plan.collect_command.index("--max_runtime_episodes") + 1] == "1000"
    assert "--allow_incomplete_episodes" not in plan.collect_command
    assert "--min_arrival_fraction" not in plan.collect_command
    assert plan.collect_command[plan.collect_command.index("--random_av_fraction") + 1] == "0.6"
    assert plan.collect_command[plan.collect_command.index("--epsilon") + 1] == "0.4"
    assert plan.collect_command[plan.collect_command.index("--safe_reroute_min_remaining_m") + 1] == "35.0"
    assert plan.collect_command[plan.collect_command.index("--completion_drain_steps") + 1] == "7200"
    assert plan.collect_command[plan.collect_command.index("--completion_drain_patience_steps") + 1] == "600"
    assert plan.collect_command[plan.collect_command.index("--sumo_time_to_teleport") + 1] == "600"
    assert plan.marl_command[plan.marl_command.index("--network_folder") + 1] == str(network.resolve())
    assert plan.marl_command[plan.marl_command.index("--network_name") + 1] == "toy_net"
    assert plan.marl_command[plan.marl_command.index("--training_family") + 1] == "dreamer_ctde"
    assert plan.marl_command[plan.marl_command.index("--vehicle_candidate_source") + 1] == "od_suffix"
    assert plan.marl_command[plan.marl_command.index("--vehicle_routes_csv") + 1] == str((network / "routes.csv").resolve())
    assert plan.dynamics_command[plan.dynamics_command.index("--action_key") + 1] == "av_planned_edge_flow"
    assert plan.dynamics_command[plan.dynamics_command.index("--action_encoder_type") + 1] == "graph_cond"
    assert "--edge_action_injection" in plan.dynamics_command
    assert plan.marl_command[plan.marl_command.index("--wm_edge_cost_source") + 1] == "decoded_speed"
    assert plan.marl_command[plan.marl_command.index("--dreamer_entropy_coef") + 1] == "0.001"
    assert plan.marl_command[plan.marl_command.index("--dreamer_action_mode") + 1] == "expected"
    assert plan.marl_command[plan.marl_command.index("--dreamer_action_temperature") + 1] == "1.0"
    assert plan.marl_command[plan.marl_command.index("--dreamer_actor_horizon") + 1] == "8"
    assert plan.marl_command[plan.marl_command.index("--seed") + 1] == "42"
    assert plan.marl_command[plan.marl_command.index("--dreamer_route_cost_reward_weight") + 1] == "0.0"
    assert plan.marl_command[plan.marl_command.index("--dreamer_grpo_route_cost_weight") + 1] == "0.05"
    assert plan.marl_command[plan.marl_command.index("--dreamer_grpo_rollout_weight") + 1] == "1.0"
    assert plan.marl_command[plan.marl_command.index("--dreamer_grpo_policy_clip_ratio") + 1] == "0.2"
    assert plan.marl_command[plan.marl_command.index("--dreamer_grpo_policy_kl_coef") + 1] == "0.0"
    assert plan.marl_command[plan.marl_command.index("--dreamer_grpo_combined_advantage_clip") + 1] == "5.0"
    assert plan.marl_command[plan.marl_command.index("--dreamer_grpo_advantage_temperature") + 1] == "60.0"
    assert plan.marl_command[plan.marl_command.index("--dreamer_grpo_planned_av_pressure_weight") + 1] == "0.0"
    assert plan.marl_command[plan.marl_command.index("--dreamer_route_cost_aux_weight") + 1] == "0.0"
    assert plan.marl_command[plan.marl_command.index("--dreamer_route_cost_aux_temperature") + 1] == "30.0"
    assert plan.marl_command[plan.marl_command.index("--dreamer_route_cost_aux_nonzero_margin") + 1] == "0.0"
    assert "--dreamer_route_cost_aux_balance" not in plan.marl_command
    assert "--dreamer_route_cost_aux_hard_target" not in plan.marl_command
    assert "--vehicle_obs_include_route_costs" not in plan.marl_command
    assert plan.eval_commands[0][plan.eval_commands[0].index("--wm_edge_cost_source") + 1] == "decoded_speed"
    assert plan.eval_commands[0][plan.eval_commands[0].index("--av_candidate_source") + 1] == "od_suffix"
    assert "--av_route_cost_policy" not in plan.eval_commands[0]
    assert plan.eval_commands[0][plan.eval_commands[0].index("--post_mutation_state_in") + 1] == str(state_in.resolve())
    assert "--keep_human_learning_after_mutation" not in plan.eval_commands[0]
    assert plan.eval_commands[0][plan.eval_commands[0].index("--policy_ckpt") + 1].endswith("checkpoint_ep00100.pt")
    assert plan.eval_commands[1][plan.eval_commands[1].index("--policy_ckpt") + 1].endswith("checkpoint_ep00200.pt")

    generated = plan.collect_settings.read_text(encoding="utf-8")
    assert f"NETWORK_DIR=\"{network.resolve()}\"" in generated
    assert "NETWORK_NAME=\"toy_net\"" in generated
    assert f"ROUTE_ARTIFACTS_DIR=\"{network.resolve()}\"" in generated
    assert "INCIDENT_EDGE=\"201201950#0\"" in generated
    assert "INCIDENT_START=\"500.0\"" in generated
    assert "INCIDENT_END=\"1500.0\"" in generated
    assert "INCIDENT_SPEED=\"1.0\"" in generated
    assert "TRAINING_FAMILY=\"dreamer_ctde\"" in generated
    assert "WM_EDGE_COST_SOURCE=\"decoded_speed\"" in generated
    assert "AV_CANDIDATE_SOURCE=\"od_suffix\"" in generated


def test_full_cycle_runner_allows_overriding_marl_actor_horizon(tmp_path):
    network = tmp_path / "network"
    _write_network(network)

    args = parse_args(
        [
            "--network-dir",
            str(network),
            "--network-name",
            "toy_net",
            "--generated-config-dir",
            str(tmp_path / "generated"),
            "--output-root",
            str(tmp_path / "data"),
            "--logs-root",
            str(tmp_path / "logs"),
            "--results-root",
            str(tmp_path / "results"),
            "--python-bin",
            "python",
            "--marl-dreamer-actor-horizon",
            "12",
        ]
    )
    plan = build_plan(args)

    assert plan.marl_command[plan.marl_command.index("--dreamer_actor_horizon") + 1] == "12"


def test_full_cycle_runner_allows_overriding_marl_seed(tmp_path):
    network = tmp_path / "network"
    _write_network(network)

    args = parse_args(
        [
            "--network-dir",
            str(network),
            "--network-name",
            "toy_net",
            "--generated-config-dir",
            str(tmp_path / "generated"),
            "--output-root",
            str(tmp_path / "data"),
            "--logs-root",
            str(tmp_path / "logs"),
            "--results-root",
            str(tmp_path / "results"),
            "--python-bin",
            "python",
            "--marl-seed",
            "9621",
        ]
    )
    plan = build_plan(args)

    assert plan.marl_command[plan.marl_command.index("--seed") + 1] == "9621"


def test_full_cycle_runner_forwards_eval_route_cost_policy(tmp_path):
    network = tmp_path / "network"
    _write_network(network)

    args = parse_args(
        [
            "--network-dir",
            str(network),
            "--network-name",
            "toy_net",
            "--generated-config-dir",
            str(tmp_path / "generated"),
            "--output-root",
            str(tmp_path / "data"),
            "--logs-root",
            str(tmp_path / "logs"),
            "--results-root",
            str(tmp_path / "results"),
            "--python-bin",
            "python",
            "--eval-av-route-cost-policy",
            "wm_current_bpr",
            "--eval-av-route-cost-length-norm",
            "sum",
            "--eval-av-route-cost-temperature",
            "0.0",
            "--eval-av-route-hdv-pressure-penalty",
            "12.5",
            "--eval-av-route-wm-live-blend-weight",
            "0.25",
            "--eval-av-route-planned-av-pressure-penalty",
            "300.0",
        ]
    )
    plan = build_plan(args)

    cmd = plan.eval_commands[0]
    assert cmd[cmd.index("--av_route_cost_policy") + 1] == "wm_current_bpr"
    assert cmd[cmd.index("--av_route_cost_length_norm") + 1] == "sum"
    assert cmd[cmd.index("--av_route_cost_temperature") + 1] == "0.0"
    assert cmd[cmd.index("--av_route_hdv_pressure_penalty") + 1] == "12.5"
    assert cmd[cmd.index("--av_route_wm_live_blend_weight") + 1] == "0.25"
    assert cmd[cmd.index("--av_route_planned_av_pressure_penalty") + 1] == "300.0"
