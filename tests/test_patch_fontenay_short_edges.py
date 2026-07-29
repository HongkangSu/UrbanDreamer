from __future__ import annotations

import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "patch_fontenay_short_edges.py"
)
SPEC = importlib.util.spec_from_file_location(
    "patch_fontenay_short_edges",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
patcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = patcher
SPEC.loader.exec_module(patcher)


def _write_network(path: Path) -> None:
    path.write_bytes(
        b'<?xml version="1.0" encoding="UTF-8"?>\r\n'
        b"<!-- formatting must be preserved -->\r\n"
        b"<net>\r\n"
        b'  <edge id="road-a" from="n0" to="n1">\r\n'
        b'    <lane id="road-a_0" index="0" speed="10" length="0.20"/>\r\n'
        b'    <lane id="road-a_1" index="1" speed="10" length="0.99"/>\r\n'
        b"  </edge>\r\n"
        b'  <edge id="road-b" from="n1" to="n2">\r\n'
        b'    <lane id="road-b_0" index="0" speed="10" length="1.00"/>\r\n'
        b"  </edge>\r\n"
        b'  <edge id=":internal" function="internal">\r\n'
        b'    <lane id=":internal_0" index="0" speed="5" length="0.10"/>\r\n'
        b"  </edge>\r\n"
        b"</net>\r\n"
    )


def test_patch_preserves_xml_and_changes_only_short_external_lanes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "fontenay.net.xml"
    output = tmp_path / "fontenay.patched.net.xml"
    _write_network(source)

    summary = patcher.patch_network(
        source,
        output_path=output,
        validate_known_counts=False,
    )

    assert summary["patched_edges"] == 1
    assert summary["patched_lanes"] == 2
    assert source.read_bytes().count(b'length="10.00"') == 0
    patched_bytes = output.read_bytes()
    assert b"<!-- formatting must be preserved -->\r\n" in patched_bytes

    root = ET.parse(output).getroot()
    lengths = {
        lane.get("id"): lane.get("length")
        for edge in root.findall("edge")
        for lane in edge.findall("lane")
    }
    assert lengths == {
        "road-a_0": "10.00",
        "road-a_1": "10.00",
        "road-b_0": "1.00",
        ":internal_0": "0.10",
    }


def test_strict_validation_rejects_an_unknown_source_without_writing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "fontenay.net.xml"
    output = tmp_path / "fontenay.patched.net.xml"
    _write_network(source)

    with pytest.raises(patcher.PatchError, match="does not match"):
        patcher.patch_network(source, output_path=output)

    assert not output.exists()
