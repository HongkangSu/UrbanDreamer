#!/usr/bin/env python3
"""Apply the Fontenay short-edge capacity patch to a SUMO network file.

The known unpatched network contains 7,585 external edges. Exactly 885 lanes
across 856 external edges have lengths below one metre. The retained
UrbanDreamer network raises those lane lengths to 10 metres without changing
edge IDs, connectivity, speeds, permissions, or lane counts.

By default the input is preserved and ``*.patched.net.xml`` is written beside
it. Use ``--in-place`` only when an automatic backup is desired.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


EXPECTED_EXTERNAL_EDGES = 7_585
EXPECTED_PATCHED_EDGES = 856
EXPECTED_PATCHED_LANES = 885
SHORT_LANE_THRESHOLD_METRES = 1.0
PATCHED_LANE_LENGTH_TEXT = "10.00"

_LANE_TAG_PATTERN = re.compile(r"<lane\b[^>]*>", re.DOTALL)
_ID_ATTRIBUTE_PATTERN = re.compile(r"\bid\s*=\s*([\"'])(.*?)\1", re.DOTALL)
_LENGTH_ATTRIBUTE_PATTERN = re.compile(
    r"(\blength\s*=\s*)([\"'])(.*?)(\2)",
    re.DOTALL,
)


class PatchError(RuntimeError):
    """Raised when the input does not match the safe patch contract."""


@dataclass(frozen=True)
class LanePatch:
    edge_id: str
    lane_id: str
    original_length_text: str


@dataclass(frozen=True)
class PatchPlan:
    external_edge_count: int
    lanes: tuple[LanePatch, ...]

    @property
    def patched_edge_count(self) -> int:
        return len({patch.edge_id for patch in self.lanes})


def build_patch_plan(network_path: Path) -> PatchPlan:
    """Inspect a SUMO network and return every external lane below one metre."""

    try:
        root = ET.parse(network_path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise PatchError(f"cannot parse SUMO network {network_path}: {exc}") from exc

    external_edge_count = 0
    lane_ids: set[str] = set()
    patches: list[LanePatch] = []

    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        if not edge_id:
            raise PatchError("encountered an edge without an id")
        if edge_id.startswith(":") or edge.get("function") == "internal":
            continue

        external_edge_count += 1
        for lane in edge.findall("lane"):
            lane_id = lane.get("id")
            length_text = lane.get("length")
            if not lane_id or length_text is None:
                raise PatchError(f"edge {edge_id!r} contains an incomplete lane")
            if lane_id in lane_ids:
                raise PatchError(f"duplicate lane id {lane_id!r}")
            lane_ids.add(lane_id)

            try:
                length_metres = float(length_text)
            except ValueError as exc:
                raise PatchError(
                    f"lane {lane_id!r} has non-numeric length {length_text!r}"
                ) from exc
            if length_metres < SHORT_LANE_THRESHOLD_METRES:
                patches.append(
                    LanePatch(
                        edge_id=edge_id,
                        lane_id=lane_id,
                        original_length_text=length_text,
                    )
                )

    return PatchPlan(
        external_edge_count=external_edge_count,
        lanes=tuple(patches),
    )


def validate_known_fontenay_plan(plan: PatchPlan) -> None:
    """Reject inputs that are not the known unpatched Fontenay network."""

    observed = (
        plan.external_edge_count,
        plan.patched_edge_count,
        len(plan.lanes),
    )
    expected = (
        EXPECTED_EXTERNAL_EDGES,
        EXPECTED_PATCHED_EDGES,
        EXPECTED_PATCHED_LANES,
    )
    if observed != expected:
        raise PatchError(
            "input does not match the known unpatched Fontenay network: "
            f"observed external_edges={observed[0]}, "
            f"short_edges={observed[1]}, short_lanes={observed[2]}; "
            f"expected {expected[0]}, {expected[1]}, {expected[2]}. "
            "Use --allow-count-mismatch only after manually verifying the source."
        )


def render_patched_xml(network_path: Path, plan: PatchPlan) -> tuple[str, str]:
    """Return the original and patched XML while preserving source formatting."""

    try:
        with network_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as stream:
            source_text = stream.read()
    except OSError as exc:
        raise PatchError(f"cannot read {network_path}: {exc}") from exc

    patch_by_lane = {patch.lane_id: patch for patch in plan.lanes}
    patched_lane_ids: set[str] = set()

    def replace_lane_tag(match: re.Match[str]) -> str:
        lane_tag = match.group(0)
        id_match = _ID_ATTRIBUTE_PATTERN.search(lane_tag)
        if id_match is None:
            return lane_tag

        lane_id = id_match.group(2)
        patch = patch_by_lane.get(lane_id)
        if patch is None:
            return lane_tag
        if lane_id in patched_lane_ids:
            raise PatchError(f"lane {lane_id!r} occurs more than once in the XML text")

        length_match = _LENGTH_ATTRIBUTE_PATTERN.search(lane_tag)
        if length_match is None:
            raise PatchError(f"lane {lane_id!r} has no textual length attribute")
        observed_length_text = length_match.group(3)
        try:
            lengths_match = float(observed_length_text) == float(
                patch.original_length_text
            )
        except ValueError:
            lengths_match = False
        if not lengths_match:
            raise PatchError(
                f"lane {lane_id!r} changed between XML inspection and rewrite: "
                f"{patch.original_length_text!r} != {observed_length_text!r}"
            )

        patched_lane_ids.add(lane_id)
        start, end = length_match.span(3)
        return (
            lane_tag[:start]
            + PATCHED_LANE_LENGTH_TEXT
            + lane_tag[end:]
        )

    patched_text = _LANE_TAG_PATTERN.sub(replace_lane_tag, source_text)
    missing_lane_ids = set(patch_by_lane) - patched_lane_ids
    if missing_lane_ids:
        examples = ", ".join(sorted(missing_lane_ids)[:5])
        raise PatchError(
            f"failed to rewrite {len(missing_lane_ids)} planned lanes; "
            f"examples: {examples}"
        )
    return source_text, patched_text


def default_output_path(network_path: Path) -> Path:
    suffix = ".net.xml"
    if network_path.name.endswith(suffix):
        base_name = network_path.name[: -len(suffix)]
        return network_path.with_name(f"{base_name}.patched{suffix}")
    return network_path.with_name(f"{network_path.name}.patched")


def patch_network(
    network_path: Path,
    *,
    output_path: Path | None = None,
    in_place: bool = False,
    dry_run: bool = False,
    validate_known_counts: bool = True,
) -> dict[str, object]:
    """Validate, patch, and safely write one Fontenay SUMO network."""

    network_path = network_path.expanduser().resolve()
    if not network_path.is_file():
        raise PatchError(f"input network is not a file: {network_path}")
    if in_place and output_path is not None:
        raise PatchError("--in-place and --output are mutually exclusive")

    plan = build_patch_plan(network_path)
    if validate_known_counts:
        validate_known_fontenay_plan(plan)
    if not plan.lanes:
        raise PatchError(
            "no external lane below one metre was found; "
            "the network may already be patched"
        )

    source_text, patched_text = render_patched_xml(network_path, plan)
    target_path = (
        network_path
        if in_place
        else (
            output_path.expanduser().resolve()
            if output_path is not None
            else default_output_path(network_path)
        )
    )
    if target_path == network_path and not in_place:
        raise PatchError("refusing to overwrite the input without --in-place")

    backup_path: Path | None = None
    if not dry_run:
        if not target_path.parent.is_dir():
            raise PatchError(f"output directory does not exist: {target_path.parent}")
        if in_place:
            backup_path = network_path.with_name(
                f"{network_path.name}.pre_short_edge_patch"
            )
            if backup_path.exists():
                raise PatchError(f"refusing to overwrite backup: {backup_path}")
            shutil.copy2(network_path, backup_path)
        elif target_path.exists():
            raise PatchError(f"refusing to overwrite existing output: {target_path}")

        _atomic_write_text(
            target_path,
            patched_text,
            mode=stat.S_IMODE(network_path.stat().st_mode),
        )

    return {
        "status": "dry_run" if dry_run else "patched",
        "input": str(network_path),
        "output": str(target_path),
        "backup": str(backup_path) if backup_path is not None else None,
        "external_edges": plan.external_edge_count,
        "patched_edges": plan.patched_edge_count,
        "patched_lanes": len(plan.lanes),
        "source_sha256": _sha256(source_text),
        "patched_sha256": _sha256(patched_text),
    }


def _atomic_write_text(path: Path, text: str, *, mode: int) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.chmod(mode)
        os.replace(temporary_path, path)
    except OSError as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise PatchError(f"cannot write patched network {path}: {exc}") from exc


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Raise all external Fontenay lane lengths below one metre to "
            "10 metres, with strict 856-edge/885-lane validation."
        )
    )
    parser.add_argument("network", type=Path, help="unpatched SUMO .net.xml file")
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "--output",
        type=Path,
        help="output path; defaults to <name>.patched.net.xml",
    )
    destination.add_argument(
        "--in-place",
        action="store_true",
        help="replace the input after creating a .pre_short_edge_patch backup",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and report changes without writing a file",
    )
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help=(
            "disable the known 7585/856/885 source fingerprint; use only after "
            "manually verifying a different Fontenay source version"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    try:
        summary = patch_network(
            args.network,
            output_path=args.output,
            in_place=args.in_place,
            dry_run=args.dry_run,
            validate_known_counts=not args.allow_count_mismatch,
        )
    except PatchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
