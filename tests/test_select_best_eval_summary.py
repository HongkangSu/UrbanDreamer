import math
import os
import sys
from pathlib import Path

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import select_best_eval_summary as select_best  # noqa: E402


def _write_summary(root: Path, *, name: str, t_all: float, t_cav: float, t_hdv: float) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    path = d / "collector_eval_summary.json"
    path.write_text(
        (
            "{\n"
            f'  "policy_ckpt": "{name}.pt",\n'
            f'  "test_eps": 3,\n'
            f'  "ratio_machines": 0.4,\n'
            f'  "t_CAV": {t_cav},\n'
            f'  "t_HDV": {t_hdv},\n'
            f'  "t_all": {t_all}\n'
            "}\n"
        ),
        encoding="utf-8",
    )
    return d


def test_build_selection_payload_ranks_by_t_all(tmp_path: Path):
    d20 = _write_summary(tmp_path, name="ep20", t_all=4.26, t_cav=4.26, t_hdv=4.26)
    d40 = _write_summary(tmp_path, name="ep40", t_all=4.25, t_cav=4.25, t_hdv=4.26)
    d60 = _write_summary(tmp_path, name="ep60", t_all=4.25, t_cav=4.25, t_hdv=4.26)

    payload = select_best.build_selection_payload([str(d20), str(d40), str(d60)])

    assert payload["config"]["metric"] == "t_all"
    assert payload["best"]["results_dir"].endswith("ep40")
    assert payload["best"]["metric_value"] == 4.25
    assert [row["rank"] for row in payload["ranked"]] == [1, 2, 3]
    assert payload["ranked"][1]["results_dir"].endswith("ep60")
    assert payload["ranked"][2]["results_dir"].endswith("ep20")


def test_resolve_summary_path_accepts_results_dir(tmp_path: Path):
    results_dir = _write_summary(tmp_path, name="probe", t_all=4.62, t_cav=5.08, t_hdv=4.33)
    path = select_best.resolve_summary_path(str(results_dir))
    assert path.name == "collector_eval_summary.json"
    assert path.exists()


def test_apply_checkpoint_decision_selects_initial_best(tmp_path: Path):
    d20 = _write_summary(tmp_path, name="ep20", t_all=4.26, t_cav=4.26, t_hdv=4.26)
    challenger = select_best.load_summary(str(d20))
    cfg = select_best.CheckpointDecisionConfig()

    decision = select_best.apply_checkpoint_decision(
        incumbent=None,
        challenger=challenger,
        config=cfg,
        patience_before=0,
    )

    assert decision["decision"] == "select_best"
    assert decision["decision_reason"] == "initial_best"
    assert decision["selected_checkpoint_after_decision"] == "ep20.pt"
    assert decision["delta_vs_best"] is None


def test_apply_checkpoint_decision_tie_keeps_earliest_and_increments_patience(tmp_path: Path):
    d40 = _write_summary(tmp_path, name="ep40", t_all=4.25, t_cav=4.25, t_hdv=4.26)
    d60 = _write_summary(tmp_path, name="ep60", t_all=4.25, t_cav=4.25, t_hdv=4.26)
    incumbent = select_best.build_summary_row(select_best.load_summary(str(d40)), metric="t_all")
    challenger = select_best.load_summary(str(d60))
    cfg = select_best.CheckpointDecisionConfig(min_delta=0.0, tie_rule="keep_incumbent_earliest")

    decision = select_best.apply_checkpoint_decision(
        incumbent=incumbent,
        challenger=challenger,
        config=cfg,
        patience_before=2,
    )

    assert decision["decision"] == "keep_best"
    assert decision["decision_reason"] == "tie_within_min_delta"
    assert decision["selected_checkpoint_after_decision"] == "ep40.pt"
    assert decision["patience_after"] == 3


def test_apply_checkpoint_decision_near_tie_uses_min_delta(tmp_path: Path):
    incumbent = select_best.build_summary_row(
        {
            "_summary_path": str(tmp_path / "incumbent.json"),
            "_results_dir": str(tmp_path),
            "policy_ckpt": "incumbent.pt",
            "test_eps": 1,
            "ratio_machines": 0.4,
            "t_CAV": 100.0,
            "t_HDV": 100.0,
            "t_all": 100.0,
        },
        metric="t_all",
    )
    challenger = {
        "_summary_path": str(tmp_path / "challenger.json"),
        "_results_dir": str(tmp_path),
        "policy_ckpt": "challenger.pt",
        "test_eps": 1,
        "ratio_machines": 0.4,
        "t_CAV": 99.7,
        "t_HDV": 99.7,
        "t_all": 99.7,
    }
    cfg = select_best.CheckpointDecisionConfig(min_delta=0.5)

    decision = select_best.apply_checkpoint_decision(
        incumbent=incumbent,
        challenger=challenger,
        config=cfg,
        patience_before=0,
    )

    assert decision["decision"] == "keep_best"
    assert decision["decision_reason"] == "tie_within_min_delta"
    assert decision["selected_checkpoint_after_decision"] == "incumbent.pt"
    assert abs(decision["delta_vs_best"]) <= 0.5
    assert decision["patience_after"] == 1


def test_apply_checkpoint_decision_strict_improvement_resets_patience(tmp_path: Path):
    incumbent = select_best.build_summary_row(
        {
            "_summary_path": str(tmp_path / "incumbent.json"),
            "_results_dir": str(tmp_path),
            "policy_ckpt": "incumbent.pt",
            "test_eps": 1,
            "ratio_machines": 0.4,
            "t_CAV": 100.0,
            "t_HDV": 100.0,
            "t_all": 100.0,
        },
        metric="t_all",
    )
    challenger = {
        "_summary_path": str(tmp_path / "challenger.json"),
        "_results_dir": str(tmp_path),
        "policy_ckpt": "challenger.pt",
        "test_eps": 1,
        "ratio_machines": 0.4,
        "t_CAV": 99.4,
        "t_HDV": 99.4,
        "t_all": 99.4,
    }
    cfg = select_best.CheckpointDecisionConfig(min_delta=0.5)

    decision = select_best.apply_checkpoint_decision(
        incumbent=incumbent,
        challenger=challenger,
        config=cfg,
        patience_before=4,
    )

    assert decision["decision"] == "select_best"
    assert decision["decision_reason"] == "strict_improvement"
    assert decision["selected_checkpoint_after_decision"] == "challenger.pt"
    assert decision["patience_after"] == 0
