import json
from pathlib import Path

import numpy as np
import torch

from ablation.online_wm_dreamer_actor_critic import CausalHdvResponsePredictor
from scripts.evaluate_hdv_response_validation import (
    CaseSpec,
    EvaluationConfig,
    _attach_prequential_accuracy,
    _distribution_metrics,
    _moving_block_bootstrap_ci,
    _normalise_distribution,
    evaluate_case,
    write_case_artifacts,
)


def test_distribution_metrics_are_zero_for_identical_distributions():
    target = np.asarray([0.1, 0.2, 0.7], dtype=np.float64)
    metrics = _distribution_metrics(target, target, top_k=2)

    assert metrics["kl"] == 0.0
    assert metrics["js"] == 0.0
    assert metrics["tv"] == 0.0
    assert metrics["topk_recall"] == 1.0


def test_normalise_distribution_matches_predictor_empty_input_semantics():
    assert np.allclose(_normalise_distribution(np.zeros(4)), np.full(4, 0.25))


def test_moving_block_bootstrap_is_deterministic():
    first = _moving_block_bootstrap_ci([1.0, -0.5, 0.25, -0.25], replicates=100, block_size=2, seed=7)
    second = _moving_block_bootstrap_ci([1.0, -0.5, 0.25, -0.25], replicates=100, block_size=2, seed=7)

    assert first == second
    assert first["mean"] == 0.125
    assert first["block_size"] == 2


def test_prequential_accuracy_filters_to_accepted_predictor_rows(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "online_wm_train_episodes.csv").write_text(
        "episode,hdv_response_pred_tv,hdv_response_pred_source_code,wm_replay_accepted\n"
        "100,0.90,2.0,1\n"
        "101,0.11,2.0,1\n"
        "102,0.12,2.0,0\n"
        "103,0.13,1.0,1\n"
        "104,nan,2.0,1\n"
        "105,0.15,2.0,1\n",
        encoding="utf-8",
    )
    episode_rows = [
        {"episode": episode, "persistence_tv": 0.05}
        for episode in range(101, 106)
    ]

    summary = _attach_prequential_accuracy(
        run_dir,
        episode_rows,
        cfg=EvaluationConfig(
            bootstrap_replicates=20,
            bootstrap_block_size=2,
        ),
    )

    assert summary["available"] is True
    assert summary["pairs"] == 2
    assert summary["episode_min"] == 101
    assert summary["episode_max"] == 105
    assert np.isclose(summary["predictor_tv"]["mean"], 0.13)
    attached_episodes = [
        row["episode"]
        for row in episode_rows
        if "online_prequential_predictor_tv" in row
    ]
    assert attached_episodes == [101, 105]


def _write_trace(trace_dir: Path, episode: int, av: np.ndarray, hdv: np.ndarray) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        trace_dir / f"episode_{episode:05d}.npz",
        episode=np.asarray([episode], dtype=np.int64),
        av_action=np.stack([av, av], axis=0).astype(np.float32),
        hdv_static_edge_demand_mean=hdv.astype(np.float32),
        hdv_target_source=np.asarray(["static_edge_demand"]),
        edge_ids=np.asarray(["e0", "e1", "e2"]),
    )


def test_evaluate_case_uses_only_future_adjacent_episode_pairs(tmp_path: Path):
    run_dir = tmp_path / "run"
    trace_dir = run_dir / "hdv_edge_pressure_traces"
    av_rows = [
        np.asarray([0.8, 0.1, 0.1]),
        np.asarray([0.7, 0.2, 0.1]),
        np.asarray([0.6, 0.3, 0.1]),
        np.asarray([0.2, 0.7, 0.1]),
        np.asarray([0.1, 0.7, 0.2]),
        np.asarray([0.1, 0.2, 0.7]),
    ]
    hdv_rows = [
        np.asarray([0.1, 0.2, 0.7]),
        np.asarray([0.1, 0.3, 0.6]),
        np.asarray([0.2, 0.3, 0.5]),
        np.asarray([0.5, 0.3, 0.2]),
        np.asarray([0.6, 0.3, 0.1]),
        np.asarray([0.7, 0.2, 0.1]),
    ]
    for episode, (av, hdv) in enumerate(zip(av_rows, hdv_rows), start=1):
        _write_trace(trace_dir, episode, av, hdv)

    (run_dir / "online_wm_dreamer_config.json").write_text(
        json.dumps(
            {
                "hdv_response_predictor_hidden_dim": 4,
                "hdv_response_predictor_depth": 1,
                "hdv_response_predictor_progress_denominator": 3,
            }
        ),
        encoding="utf-8",
    )
    model = CausalHdvResponsePredictor(n_edges=3, hidden_dim=4, depth=1)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    checkpoint = run_dir / "checkpoint_ep00003.pt"
    torch.save(
        {
            "episode": 3,
            "hdv_response_predictor": model.state_dict(),
            "hdv_response_trained_pairs": 2,
        },
        checkpoint,
    )
    (run_dir / "online_wm_train_episodes.csv").write_text(
        "episode,hdv_response_pred_tv,hdv_response_pred_source_code,wm_replay_accepted\n"
        "4,0.30,2.0,1\n"
        "5,0.20,2.0,1\n"
        "6,0.10,2.0,1\n",
        encoding="utf-8",
    )

    result = evaluate_case(
        CaseSpec(name="tiny", run_dir=run_dir, checkpoint=checkpoint),
        EvaluationConfig(
            top_k=2,
            match_pool_size=2,
            bootstrap_replicates=50,
            bootstrap_block_size=2,
            hash_checkpoint=False,
        ),
    )

    assert [row["episode"] for row in result.episode_rows] == [4, 5, 6]
    assert result.summary["split"]["test_pairs"] == 3
    assert result.summary["split"]["test_episode_min"] == 4
    assert result.summary["provenance"]["checkpoint_episode"] == 3
    assert result.summary["provenance"]["checkpoint_trained_pairs"] == 2
    assert len(result.counterfactual_rows) == 3
    assert result.summary["accuracy"]["predictor"]["js"]["n"] == 3
    assert result.summary["adaptation_events"]["changed_pairs"] == 3
    assert result.summary["adaptation_events"]["unchanged_pairs"] == 0
    prequential = result.summary["online_prequential_accuracy"]
    assert prequential["available"] is True
    assert prequential["pairs"] == 3
    assert np.isclose(prequential["predictor_tv"]["mean"], 0.2)
    assert all("online_prequential_predictor_tv" in row for row in result.episode_rows)

    output_dir = tmp_path / "output"
    write_case_artifacts(result, output_dir)
    assert (output_dir / "tiny" / "summary.json").exists()
    assert (output_dir / "tiny" / "episode_metrics.csv").exists()
    assert (output_dir / "tiny" / "counterfactual_interventions.csv").exists()
    assert (output_dir / "tiny" / "representative_field.csv").exists()
