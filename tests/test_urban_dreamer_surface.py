from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import urban_dreamer


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "urban_dreamer"


def test_repository_root_contains_the_complete_release_surface() -> None:
    required_paths = {
        ".gitattributes",
        "README.md",
        "requirements.txt",
        "environment.yaml",
        "configs/mainline.example.toml",
    }
    missing = sorted(path for path in required_paths if not (ROOT / path).is_file())
    assert missing == []


def test_city_configs_use_repository_network_bundles() -> None:
    city_names = (
        "toy_cross_bottleneck",
        "ingolstadt_custom",
        "gretz_armainvilliers",
        "fontenay_en_parisis",
    )
    for city_name in city_names:
        config_path = ROOT / "configs" / f"{city_name}.toml"
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        inputs = config["inputs"]
        bundle_path = f"../networks/{city_name}"

        assert inputs["network_name"] == city_name
        assert inputs["network_dir"] == bundle_path
        assert set(inputs) == {"network_name", "network_dir"}
        assert config["simulation"]["vehicle_max_speed_mps"] == 40.0
        assert config["run"]["output_dir"] == f"../outputs/{city_name}"


def test_package_exposes_only_the_supported_mainline_api() -> None:
    assert urban_dreamer.__all__ == ["load_config", "run_mainline"]
    assert callable(urban_dreamer.load_config)
    assert callable(urban_dreamer.run_mainline)


def test_mainline_has_one_small_semantic_module_tree() -> None:
    assert {
        path.relative_to(PACKAGE).as_posix()
        for path in PACKAGE.rglob("*.py")
    } == {
        "__init__.py",
        "__main__.py",
        "learning/__init__.py",
        "learning/baselines/__init__.py",
        "learning/baselines/common.py",
        "learning/baselines/dreamer.py",
        "learning/baselines/dta.py",
        "learning/baselines/gnac.py",
        "learning/baselines/mbpo.py",
        "learning/observations.py",
        "learning/episode_archive.py",
        "learning/imagination.py",
        "learning/replay.py",
        "learning/training.py",
        "models/__init__.py",
        "models/policy.py",
        "models/world_model.py",
        "traffic/__init__.py",
        "traffic/demand.py",
        "traffic/environment.py",
        "traffic/event_session.py",
        "traffic/routing.py",
        "traffic/topology.py",
        "workflow/__init__.py",
        "workflow/checkpoint.py",
        "workflow/config.py",
        "workflow/pipeline.py",
    }


def test_mainline_never_imports_legacy_or_ignored_branches() -> None:
    forbidden_roots = {"dreamer4", "scripts", "routerl", "routerlurb"}
    offenders: list[str] = []
    for source_path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"),
            filename=str(source_path),
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".", 1)[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots = {node.module.split(".", 1)[0]}
            else:
                continue
            forbidden = roots & forbidden_roots
            if forbidden:
                offenders.append(
                    f"{source_path.relative_to(PACKAGE)}:"
                    f"{node.lineno}:{sorted(forbidden)}"
                )
    assert offenders == []


def test_mainline_exports_only_supported_environment_surface() -> None:
    environment_path = PACKAGE / "traffic" / "environment.py"
    tree = ast.parse(
        environment_path.read_text(encoding="utf-8"),
        filename=str(environment_path),
    )
    exports: set[str] | None = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if node.targets[0].id != "__all__":
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            raise AssertionError("__all__ must stay literal for surface checks")
        exports = {
            element.value
            for element in node.value.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        }
    assert exports is not None
    assert {"NextHopDecision", "SumoSession", "EnvironmentContractError", "build_session_factory"} <= exports
    assert not {"ActionDecision", "CandidatePathGenerator", "RouteRLSessionSpec", "RouteRLSumoSession"} & exports
