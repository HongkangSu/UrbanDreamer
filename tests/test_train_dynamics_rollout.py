"""Focused rollout helper tests for train_dynamics."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dreamer4"))
from flow_dataset import FlowDataset  # noqa: E402
from train_dynamics import (  # noqa: E402
    _build_adm_action_windows,
    _build_graph_cond_model_actions,
    _build_model_actions,
    _build_adm_context_target_windows,
    _build_rollout_inputs,
    _build_rollout_targets,
    _parse_adm_horizons,
    _scheduled_sampling_prob,
    _select_rollout_state,
    cfm_predict_one_timestep,
)



def test_scheduled_sampling_probability_warms_up_linearly():
    assert _scheduled_sampling_prob(step=0, enabled=True, warmup=100, end=0.5) == 0.0
    assert _scheduled_sampling_prob(step=50, enabled=True, warmup=100, end=0.5) == 0.25
    assert _scheduled_sampling_prob(step=100, enabled=True, warmup=100, end=0.5) == 0.5
    assert _scheduled_sampling_prob(step=1000, enabled=True, warmup=100, end=0.5) == 0.5
    assert _scheduled_sampling_prob(step=50, enabled=False, warmup=100, end=0.5) == 0.0



def test_build_rollout_targets_returns_future_suffixes():
    z = torch.arange(2 * 6 * 3 * 4, dtype=torch.float32).view(2, 6, 3, 4)
    targets = _build_rollout_targets(z, horizon=3)

    assert len(targets) == 3
    assert targets[0].shape == (2, 3, 3, 4)
    assert torch.equal(targets[0], z[:, 1:4])
    assert torch.equal(targets[1], z[:, 2:5])
    assert torch.equal(targets[2], z[:, 3:6])



def test_build_rollout_inputs_aligns_actions_depart_context_by_horizon():
    act = torch.randn(2, 6, 7)
    dep = torch.randn(2, 6, 5)
    ctx = torch.randn(2, 6, 3)

    inputs = _build_rollout_inputs(act, dep, ctx, horizon=3)

    assert len(inputs) == 3
    assert inputs[0]["act"].shape == (2, 3, 7)
    assert inputs[1]["depart"].shape == (2, 3, 5)
    assert inputs[2]["context"].shape == (2, 3, 3)


def test_parse_adm_horizons_sorts_and_deduplicates_csv():
    assert _parse_adm_horizons("8,2,4,2") == (2, 4, 8)
    assert _parse_adm_horizons("") == ()


def test_build_adm_context_target_windows_repeats_start_and_targets_k_step():
    z = torch.arange(1 * 6 * 2 * 1, dtype=torch.float32).view(1, 6, 2, 1)
    ctx, target = _build_adm_context_target_windows(z, horizon=3)

    assert ctx.shape == (3, 3, 2, 1)
    assert target.shape == (3, 1, 2, 1)
    assert torch.equal(ctx[0], z[0, 0].expand(3, 2, 1))
    assert torch.equal(target[:, 0], z[0, 3:6])


def test_build_adm_action_windows_keeps_causal_actions_and_repeats_last_for_final_token():
    act = torch.arange(1 * 6 * 1, dtype=torch.float32).view(1, 6, 1)
    win = _build_adm_action_windows(act, horizon=3)

    assert win.shape == (3, 4, 1)
    assert torch.equal(win[0, :, 0], torch.tensor([0.0, 1.0, 2.0, 2.0]))
    assert torch.equal(win[2, :, 0], torch.tensor([2.0, 3.0, 4.0, 4.0]))



def test_select_rollout_state_uses_prediction_when_coin_flip_is_true():
    z_gt = torch.ones(2, 3, 4)
    z_pred = torch.zeros(2, 3, 4)
    chosen = _select_rollout_state(
        z_gt=z_gt,
        z_pred=z_pred,
        use_pred_mask=torch.tensor([True, True]),
    )
    assert torch.equal(chosen, z_pred)


def test_flow_dataset_keeps_actions_aligned_with_current_decision_step(tmp_path):
    task = "toy"
    flow_dir = tmp_path / "flows" / task
    flow_dir.mkdir(parents=True)

    flows = torch.zeros(5, 2, 8)
    flows[:, 0, 0] = torch.arange(5, dtype=torch.float32)
    torch.save({"flows": flows}, flow_dir / "shard_0000.pt")
    torch.save(
        {
            "episode": torch.zeros(5, dtype=torch.int64),
            "av_planned_edge_flow": torch.arange(10, dtype=torch.float32).view(5, 2),
            "reward": -torch.ones(5, dtype=torch.float32),
        },
        tmp_path / f"{task}.pt",
    )

    dataset = FlowDataset(
        data_dirs=str(tmp_path),
        task_name=task,
        seq_len=3,
        shard_size=5,
        verbose=False,
        reward_key="reward",
    )

    flow_b, act_b, reward_b = dataset[0]
    assert torch.equal(flow_b[:, 0, 0], torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(act_b[:, 0], torch.tensor([0.0, 2.0, 4.0]))
    assert torch.equal(reward_b, -torch.ones(3))

    model_actions = _build_model_actions(act_raw=act_b.unsqueeze(0))
    assert torch.equal(model_actions[0, :, 0], torch.tensor([0.0, 0.0, 2.0]))


def test_flow_dataset_episode_mean_aux_target_repeats_full_episode_response(tmp_path):
    task = "toy"
    flow_dir = tmp_path / "flows" / task
    flow_dir.mkdir(parents=True)

    flows = torch.zeros(6, 2, 8)
    torch.save({"flows": flows}, flow_dir / "shard_0000.pt")
    hdv = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 2.0],
            [2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    torch.save(
        {
            "episode": torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64),
            "av_planned_edge_flow": torch.zeros(6, 2),
            "hdv_edge_action": hdv,
        },
        tmp_path / f"{task}.pt",
    )

    dataset = FlowDataset(
        data_dirs=str(tmp_path),
        task_name=task,
        seq_len=3,
        shard_size=6,
        verbose=False,
        aux_action_key="hdv_edge_action",
        aux_target_mode="episode_mean",
    )

    _, _, aux_ep0 = dataset[0]
    _, _, aux_ep1 = dataset[1]
    assert torch.allclose(aux_ep0, torch.tensor([[2.0 / 3.0, 1.0 / 3.0]]).expand(3, 2))
    assert torch.allclose(aux_ep1, torch.tensor([[1.0 / 3.0, 2.0 / 3.0]]).expand(3, 2))


def test_flow_dataset_episode_action_mean_is_raw_full_episode_pressure(tmp_path):
    task = "toy"
    flow_dir = tmp_path / "flows" / task
    flow_dir.mkdir(parents=True)

    flows = torch.zeros(6, 2, 8)
    torch.save({"flows": flows}, flow_dir / "shard_0000.pt")
    act = torch.tensor(
        [
            [0.0, 1.0],
            [2.0, 3.0],
            [4.0, 5.0],
            [10.0, 0.0],
            [20.0, 0.0],
            [30.0, 0.0],
        ],
        dtype=torch.float32,
    )
    torch.save(
        {
            "episode": torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64),
            "av_planned_edge_flow": act,
            "reward": -torch.ones(6, dtype=torch.float32),
        },
        tmp_path / f"{task}.pt",
    )

    dataset = FlowDataset(
        data_dirs=str(tmp_path),
        task_name=task,
        seq_len=3,
        shard_size=6,
        verbose=False,
        reward_key="reward",
        return_episode_action_mean=True,
    )

    _, act_ep0, reward_ep0, ep_act_ep0 = dataset[0]
    _, act_ep1, reward_ep1, ep_act_ep1 = dataset[1]
    assert torch.equal(act_ep0, act[:3])
    assert torch.equal(reward_ep0, -torch.ones(3))
    assert torch.equal(act_ep1, act[3:])
    assert torch.equal(reward_ep1, -torch.ones(3))
    assert torch.allclose(ep_act_ep0, torch.tensor([[2.0, 3.0]]).expand(3, 2))
    assert torch.allclose(ep_act_ep1, torch.tensor([[20.0, 0.0]]).expand(3, 2))


def test_build_graph_cond_model_actions_shifts_each_conditioning_stream():
    act = torch.arange(1 * 4 * 2, dtype=torch.float32).view(1, 4, 2)
    dep = act + 100.0
    ctx = torch.arange(1 * 4 * 3, dtype=torch.float32).view(1, 4, 3) + 200.0

    act_s, dep_s, ctx_s = _build_graph_cond_model_actions(
        act_raw=act,
        depart_b=dep,
        context_b=ctx,
    )

    assert torch.equal(act_s[:, 0], torch.zeros_like(act[:, 0]))
    assert torch.equal(dep_s[:, 0], torch.zeros_like(dep[:, 0]))
    assert torch.equal(ctx_s[:, 0], torch.zeros_like(ctx[:, 0]))
    assert torch.equal(act_s[:, 1:], act[:, :-1])
    assert torch.equal(dep_s[:, 1:], dep[:, :-1])
    assert torch.equal(ctx_s[:, 1:], ctx[:, :-1])


def test_cfm_predict_one_timestep_slices_graph_cond_action_tuple():
    class _Dyn:
        def __call__(
            self,
            actions,
            _step_idxs,
            _signal_idxs,
            packed_seq,
            *,
            act_mask=None,
            agent_tokens=None,
            flow_time=None,
        ):
            assert isinstance(actions, tuple)
            assert actions[0].shape[:2] == packed_seq.shape[:2]
            assert actions[1].shape[:2] == packed_seq.shape[:2]
            assert actions[2].shape[:2] == packed_seq.shape[:2]
            return torch.zeros_like(packed_seq), None

    past = torch.randn(2, 3, 4, 5)
    actions = (
        torch.randn(2, 5, 4),
        torch.randn(2, 5, 4),
        torch.randn(2, 5, 3),
    )

    out = cfm_predict_one_timestep(
        _Dyn(),
        past_packed=past,
        actions=actions,
        n_steps=1,
        initial_state=torch.ones(2, 1, 4, 5),
    )

    assert torch.equal(out, torch.ones(2, 4, 5))
