from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .workflow.config import MainlineConfig, load_config
from .workflow.pipeline import run_mainline


def main(argv: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    config = _apply_overrides(
        load_config(args.config),
        output_dir=args.output_dir,
        device=args.device,
    )
    result = run_mainline(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m urban_dreamer",
        description="Run or strictly resume the online world-model mainline.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="strict mainline TOML file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="override [run].output_dir",
    )
    parser.add_argument(
        "--device",
        help="override [run].device (for example cpu or cuda:0)",
    )
    return parser


def _apply_overrides(
    config: MainlineConfig,
    *,
    output_dir: Path | None,
    device: str | None,
) -> MainlineConfig:
    run = config.run
    if output_dir is not None:
        run = replace(run, output_dir=output_dir.expanduser().resolve())
    if device is not None:
        normalized_device = device.strip()
        if not normalized_device:
            raise ValueError("--device must be non-empty")
        run = replace(run, device=normalized_device)
    return replace(config, run=run)


if __name__ == "__main__":
    raise SystemExit(main())
