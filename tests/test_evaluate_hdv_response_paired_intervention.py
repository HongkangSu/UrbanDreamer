import csv
import json
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate_hdv_response_paired_intervention import (
    _decision_summary,
    _episode_hdv_distribution,
    _load_av_pressure,
    _load_edge_ids,
    _load_routes,
    _require_matching_artifact,
    _signed_effect_error,
    _top_effect_rows,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_reconstructs_realized_hdv_edge_distribution(tmp_path: Path):
    edge_registry = tmp_path / "edge_registry.json"
    edge_registry.write_text(
        json.dumps({"edge_ids": ["e0", "e1", "e2"]}),
        encoding="utf-8",
    )
    routes_csv = tmp_path / "routes.csv"
    _write_csv(
        routes_csv,
        ["origins", "destinations", "path"],
        [
            {"origins": 0, "destinations": 1, "path": "e0 e1 e1"},
            {"origins": 0, "destinations": 1, "path": "e0 e2"},
        ],
    )
    episode_csv = tmp_path / "ep201.csv"
    _write_csv(
        episode_csv,
        [
            "id",
            "hdv_action",
            "seen_in_episode",
            "origin",
            "destination",
            "vType",
        ],
        [
            {
                "id": 1,
                "hdv_action": 0,
                "seen_in_episode": 1,
                "origin": 0,
                "destination": 1,
                "vType": "Human",
            },
            {
                "id": 2,
                "hdv_action": 1,
                "seen_in_episode": 1,
                "origin": 0,
                "destination": 1,
                "vType": "HDV",
            },
            {
                "id": 3,
                "hdv_action": 1,
                "seen_in_episode": 0,
                "origin": 0,
                "destination": 1,
                "vType": "Human",
            },
            {
                "id": 4,
                "hdv_action": 1,
                "seen_in_episode": 1,
                "origin": 0,
                "destination": 1,
                "vType": "AV",
            },
        ],
    )

    edge_ids = _load_edge_ids(edge_registry)
    distribution, diagnostics = _episode_hdv_distribution(
        episode_csv,
        routes=_load_routes(routes_csv),
        edge_to_index={edge_id: index for index, edge_id in enumerate(edge_ids)},
    )

    assert edge_ids == ["e0", "e1", "e2"]
    assert np.allclose(distribution, [0.4, 0.4, 0.2])
    assert diagnostics["hdv_vehicles"] == 2
    assert diagnostics["missing_route_assignments"] == 0
    assert diagnostics["route_counts"] == {"0": 1, "1": 1}


def test_loads_exact_episode_mean_av_pressure(tmp_path: Path):
    trace_path = tmp_path / "trace.pt"
    torch.save(
        {
            "episode": torch.tensor([201, 201, 202]),
            "av_planned_edge_flow": torch.tensor(
                [
                    [0.8, 0.2, 0.0],
                    [0.4, 0.4, 0.2],
                    [0.0, 0.0, 1.0],
                ]
            ),
        },
        trace_path,
    )

    pressure, steps = _load_av_pressure(trace_path, episode=201, n_edges=3)

    assert steps == 2
    assert np.allclose(pressure, [0.6, 0.3, 0.1])


def test_summarizes_applied_counterfactual_decisions(tmp_path: Path):
    decisions = tmp_path / "dynamic_av_decisions.csv"
    _write_csv(
        decisions,
        [
            "episode",
            "active_av",
            "counterfactual_applied",
            "counterfactual_mode",
            "selected_count_route0",
            "selected_count_route1",
            "selected_count_route2",
            "selected_count_route3",
        ],
        [
            {
                "episode": 201,
                "active_av": 3,
                "counterfactual_applied": 3,
                "counterfactual_mode": "od_override",
                "selected_count_route0": 3,
                "selected_count_route1": 0,
                "selected_count_route2": 0,
                "selected_count_route3": 0,
            },
            {
                "episode": 202,
                "active_av": 2,
                "counterfactual_applied": 0,
                "counterfactual_mode": "none",
                "selected_count_route0": 1,
                "selected_count_route1": 1,
                "selected_count_route2": 0,
                "selected_count_route3": 0,
            },
        ],
    )

    summary = _decision_summary(decisions, episode=201)

    assert summary["active_av_decisions"] == 3
    assert summary["counterfactual_application_fraction"] == 1.0
    assert summary["counterfactual_modes"] == ["od_override"]
    assert summary["selected_route_fractions"]["route0"] == 1.0


def test_signed_effect_error_is_half_l1():
    predicted = np.asarray([0.2, -0.1, -0.1])
    observed = np.asarray([0.1, -0.1, 0.0])

    assert np.isclose(_signed_effect_error(predicted, observed), 0.1)


def test_top_effect_rows_rank_predicted_mismatch_when_observed_effect_is_zero():
    rows = _top_effect_rows(
        ["e0", "e1", "e2"],
        observed_effect=np.zeros(3),
        predicted_effect=np.asarray([0.01, -0.03, 0.02]),
        baseline_before=np.asarray([0.2, 0.3, 0.5]),
        baseline_after=np.asarray([0.2, 0.3, 0.5]),
        treatment_after=np.asarray([0.2, 0.3, 0.5]),
        top_k=2,
    )

    assert [row["edge_id"] for row in rows] == ["e1", "e2"]
    assert np.isclose(rows[0]["ranking_abs_max_effect"], 0.03)


def test_matching_control_artifact_requires_identical_content(tmp_path: Path):
    baseline = tmp_path / "baseline"
    treatment = tmp_path / "treatment"
    baseline.mkdir()
    treatment.mkdir()
    (baseline / "agents.csv").write_text("id\n1\n", encoding="utf-8")
    (treatment / "agents.csv").write_text("id\n1\n", encoding="utf-8")

    digest = _require_matching_artifact(baseline, treatment, "agents.csv")

    assert len(digest) == 64
