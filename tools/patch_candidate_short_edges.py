"""Give the two candidate networks a10m external-lane length floor."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import stat
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.patch_fontenay_short_edges import (
    LanePatch,PatchPlan,render_patched_xml,_atomic_write_text,
)

MINIMUM_LENGTH_M=10.0
CITIES=('coulommiers','montereau_fault_yonne')


def patch_external_lanes(path: Path) -> dict:
    """Patch only external lane lengths, preserving the rest of the XML text."""
    root=ET.parse(path).getroot();changes=[];edges=0
    for edge in root.findall('edge'):
        if edge.get('id','').startswith(':') or edge.get('function')=='internal':
            continue
        edges+=1
        for lane in edge.findall('lane'):
            old=lane.attrib['length'];length=float(old)
            if not math.isfinite(length) or length<=0:
                raise ValueError(f'invalid lane length: {lane.attrib["id"]}={old}')
            if length<MINIMUM_LENGTH_M:
                changes.append(LanePatch(edge.attrib['id'],lane.attrib['id'],old))
    original,patched=render_patched_xml(path,PatchPlan(edges,tuple(changes)))
    if changes:
        _atomic_write_text(path,patched,mode=stat.S_IMODE(path.stat().st_mode))
    return dict(minimum_external_lane_length_m=MINIMUM_LENGTH_M,external_edges=edges,
        patched_edges=len({c.edge_id for c in changes}),patched_lanes=len(changes),
        original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        patched_sha256=hashlib.sha256(patched.encode()).hexdigest(),
        changes=[dict(edge=c.edge_id,lane=c.lane_id,old_length_m=float(c.original_length_text),
                      new_length_m=MINIMUM_LENGTH_M) for c in changes])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,
                        default=ROOT/'outputs/candidate_short_edges_10m_20261007')
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    from tools.prepare_maule_nangis import registry,sha
    from urban_dreamer.workflow.config import load_config
    from urban_dreamer.traffic.routing import validate_prepared_inputs
    from urban_dreamer.traffic.topology import load_traffic_topology
    report={'minimum_external_lane_length_m':MINIMUM_LENGTH_M,'networks':{}}
    for city in CITIES:
        directory=ROOT/'networks'/city;backup=out/'original_inputs'/city
        backup.mkdir(parents=True)
        filenames=[f'{city}.net.xml','edge_registry.json','agents.csv',f'od_{city}.txt']
        for name in filenames:
            shutil.copy2(directory/name,backup/name)
        # Historical lower-load diagnostics used links to the active inputs.
        # Pin these links to their original XML/registry before changing them.
        pinned=[]
        probes=ROOT/'outputs/ingo_reference_reselection_20261007/load_probes'
        for folder in probes.glob(f'{city}_*/inputs'):
            for name in filenames:
                link=folder/name
                if link.is_symlink() and link.resolve()==(directory/name).resolve():
                    link.unlink();link.symlink_to(backup/name);pinned.append(str(link))
        change=patch_external_lanes(directory/f'{city}.net.xml')
        reg=registry(directory/f'{city}.net.xml',city)
        _atomic_write_text(directory/'edge_registry.json',json.dumps(reg,separators=(',',':'))+'\n',
                           mode=stat.S_IMODE((directory/'edge_registry.json').stat().st_mode))
        for suffix in ('','_offline'):
            cfg=load_config(ROOT/'configs'/f'{city}{suffix}.toml')
            prepared=validate_prepared_inputs(cfg);topology=load_traffic_topology(prepared)
            assert min(topology.network.external_lane_lengths_m.values())>=MINIMUM_LENGTH_M
            assert min(topology.network.lengths_m)>=MINIMUM_LENGTH_M
            assert cfg.simulation.incident_start_seconds>cfg.simulation.evaluation_horizon_seconds
        assert all(sha(directory/name)==sha(backup/name) for name in ('agents.csv',f'od_{city}.txt'))
        parent=ROOT.parent/'networks'/city/f'{city}.net.xml'
        assert sha(parent)==sha(backup/f'{city}.net.xml')
        report['networks'][city]=dict(**change,backup=str(backup),
            expected_vehicles=prepared.vehicle_count,input_hashes=prepared.input_hashes,
            all_od_reachable=True,minimum_length_verified=True,
            parent_original_unchanged=True,pinned_historical_links=pinned)
        print(json.dumps({k:v for k,v in report['networks'][city].items()
                          if k not in ('changes','pinned_historical_links')}),flush=True)
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
