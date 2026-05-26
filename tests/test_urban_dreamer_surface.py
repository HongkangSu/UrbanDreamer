import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dreamer4.urban_routing.train_marl import build_arg_parser  # noqa: E402


def _source_texts():
    for root in (ROOT / "scripts", ROOT / "dreamer4"):
        for path in root.rglob("*.py"):
            yield path, path.read_text(encoding="utf-8")


def test_no_fixed_route_public_eval_script():
    assert not (ROOT / "scripts" / "eval_urb_fair_fixed_routes.py").exists()
    public_parser_text = "\n".join(
        (ROOT / "scripts" / name).read_text(encoding="utf-8")
        for name in ("collect_urb_fair.py", "eval_sumo.py")
    )
    assert 'add_argument("--fixed_av_route' not in public_parser_text


def test_no_external_urb_metrics_dependency_or_source_repo_path():
    forbidden = [
        "../URB",
        "BenchmarkMetrics.csv",
        "VectorMetrics.csv",
        "/home/wangwenrui/shk/dreamer4",
    ]
    offenders = []
    for path, text in _source_texts():
        for needle in forbidden:
            if needle in text:
                offenders.append(f"{path.relative_to(ROOT)}:{needle}")
    assert offenders == []


def test_no_public_legacy_training_family():
    parser = build_arg_parser()
    action = next(action for action in parser._actions if action.dest == "training_family")
    assert action.choices == ["dreamer_ctde"]
    assert "ppo_legacy" not in "\n".join(text for _, text in _source_texts())
