"""Dataset serialization helpers for SUMO evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def _save_collected_flow_dataset(
    output_dir: str,
    task_name: str,
    shard_size: int,
    *,
    episodes: list[int],
    av_edge_actions: list[np.ndarray],
    hdv_edge_actions: list[np.ndarray],
    joint_edge_actions: list[np.ndarray],
    rewards: list[float],
    flows: list[np.ndarray],
    departs: list[np.ndarray] | None = None,
    contexts: list[np.ndarray] | None = None,
    decision_indices: list[int] | None = None,
    sim_steps: list[int] | None = None,
) -> None:
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    av_edge_action_tensor = torch.from_numpy(np.stack(av_edge_actions, axis=0)).float()
    hdv_edge_action_tensor = torch.from_numpy(np.stack(hdv_edge_actions, axis=0)).float()
    joint_edge_action_tensor = torch.from_numpy(np.stack(joint_edge_actions, axis=0)).float()
    demo = {
        "episode": torch.tensor(episodes, dtype=torch.int64),
        "av_edge_action": av_edge_action_tensor,
        "av_planned_edge_flow": av_edge_action_tensor,
        "hdv_edge_action": hdv_edge_action_tensor,
        "joint_edge_action": joint_edge_action_tensor,
        "reward": torch.tensor(rewards, dtype=torch.float32),
        "action_schema_version": torch.tensor(2, dtype=torch.int64),
    }
    if decision_indices and len(decision_indices) == len(episodes):
        demo["decision_idx"] = torch.tensor(decision_indices, dtype=torch.int64)
    if sim_steps and len(sim_steps) == len(episodes):
        demo["sim_step"] = torch.tensor(sim_steps, dtype=torch.int64)
    if departs and len(departs) > 0:
        demo["depart"] = torch.from_numpy(np.stack(departs, axis=0)).float()
    if contexts and len(contexts) > 0:
        demo["context"] = torch.from_numpy(np.stack(contexts, axis=0)).float()
    demo_path = out_dir / f"{task_name}.pt"
    torch.save(demo, demo_path)

    flows_dir = out_dir / "flows" / task_name
    flows_dir.mkdir(parents=True, exist_ok=True)
    num_shards = (len(flows) + int(shard_size) - 1) // int(shard_size)
    for si in range(num_shards):
        s = si * int(shard_size)
        e = min((si + 1) * int(shard_size), len(flows))
        arr = np.stack(flows[s:e], axis=0)
        torch.save({"flows": torch.from_numpy(arr).float()}, flows_dir / f"shard_{si:04d}.pt")

    print(f"\n  Collected dataset saved:")
    print(f"    demo:  {demo_path}")
    print(f"    flows: {flows_dir} ({num_shards} shards)")
    print(f"    steps: {len(episodes)}")
