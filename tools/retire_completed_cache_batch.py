"""Publish a verified, explicitly bounded batch from completed conversion receipts.

Never deletes raw files. Parent conversion artifacts are immutable inputs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import retire_cached_trajectories as retire
from tools.build_standalone_corpus import publish_verification, old_entries
from urban_dreamer.learning.offline_data import _identity


def parent_binding(parent, row, parent_plan):
    receipt=parent/'receipts'/f"{_identity({'census_path':row['path']})}.json"
    digest=retire.sha(receipt)
    entry=retire.read(receipt)
    if retire.sha(receipt)!=digest:raise ValueError('parent_receipt_changed_during_read')
    if entry.get('plan_identity')!=parent_plan['identity'] or entry.get('census_row')!=row:
        raise ValueError('parent_receipt_binding_mismatch')
    return entry,dict(path=str(receipt.absolute()),sha256=digest,plan_identity=parent_plan['identity'])


def validation_anchors(parent_plan):
    """Keep the exact two raw sources used by the parent city's final batch check."""
    minima={}
    for row in parent_plan['rows']:
        reused=bool(row.get('reusable_cache'))
        if reused:
            entry=old_entries(row['cache_manifest'])[str(Path(row['reusable_cache']).resolve())]
            path=entry['path']
        else:path=row['path']
        if reused not in minima or path<minima[reused]:minima[reused]=path
    return sorted(minima.values())


def select_entries(parent, parent_plan, protection, *, limit=1000, local_device=None):
    candidates=[]
    for row in parent_plan['rows']:
        key=_identity({'census_path':row['path']})
        if not (parent/'receipts'/f'{key}.json').exists():continue
        paths=[]
        for name in retire.candidate_paths({'path':row['path']},row):
            if row.get('kind')!='raw' and name==row['path']:continue
            try:
                lexical=Path(name)
                if str(lexical) in protection['paths'] or retire.within(lexical,protection['roots']):continue
                path=retire.safe_path(name,retire.ALLOWED_ROOTS)
                if path.is_symlink() or retire.protected_reason(path,protection):continue
                st=path.stat()
                if local_device is not None and st.st_dev!=local_device:continue
                paths.append((st.st_blocks*512,str(path)))
            except (OSError,ValueError):continue
        if paths:candidates.append((sum(x[0] for x in paths),row,paths))
    candidates.sort(key=lambda x:(-x[0],x[1]['path']))
    selected=candidates[:limit]
    return selected,dict(eligible_entries=len(candidates),eligible_path_allocated_bytes=sum(x[0] for x in candidates),selected_entries=len(selected),selected_path_allocated_bytes=sum(x[0] for x in selected),note='Path byte sum is an upper bound; hardlink release is measured by deletion ledger.')


def publish_batch(parent, selected_rows, output, *, verify=publish_verification):
    parent=Path(parent);output=Path(output)
    if (output/'plan.json').exists():raise ValueError('batch_already_exists')
    pp=parent/'plan.json';parent_plan=retire.read(pp)
    if parent_plan['identity']!=_identity({k:v for k,v in parent_plan.items() if k!='identity'}):raise ValueError('parent_plan_identity_mismatch')
    original={r['path']:r for r in parent_plan['rows']}
    if not selected_rows or len({r['path'] for r in selected_rows})!=len(selected_rows):raise ValueError('empty_or_duplicate_batch')
    bound=[]
    for row in selected_rows:
        if original.get(row['path'])!=row:raise ValueError('row_not_in_parent_plan')
        entry,proof=parent_binding(parent,row,parent_plan)
        retire.verify_cache_and_log(entry)
        bound.append((entry,proof))
    output.mkdir(parents=True,exist_ok=True);(output/'receipts').mkdir()
    canonical=output/'canonical.jsonl'
    canonical.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in selected_rows))
    parent_proof=dict(path=str(pp.absolute()),sha256=retire.sha(pp),identity=parent_plan['identity'])
    plan=dict(city=parent_plan['city'],contract=parent_plan['contract'],scope='completed_receipt_batch',parent_plan=parent_proof,
              rows=selected_rows,limit=0,census_sha256=retire.sha(canonical),selected_count=len(selected_rows),
              completeness='complete for this explicit batch only; parent city conversion may be incomplete')
    plan['identity']=_identity(plan);retire.atomic_json(output/'plan.json',plan)
    entries=[]
    for old,proof in bound:
        entry=dict(old,plan_identity=plan['identity'],parent_receipt=proof)
        key=_identity({'census_path':entry['census_row']['path']})
        retire.atomic_json(output/'receipts'/f'{key}.json',entry);entries.append(entry)
    manifest=dict(schema_version=1,status='complete',source_storage='compact_only_v1',scope='completed_receipt_batch',
                  contract=plan['contract'],entries=entries,sources=entries,duplicate_files=0,plan_identity=plan['identity'],
                  train_episodes=sum(e['split']=='train' for e in entries),validation_episodes=sum(e['split']=='validation' for e in entries),parent_plan=parent_proof)
    manifest['identity']=_identity(manifest);retire.atomic_json(output/'cache_manifest.json',manifest)
    verify(manifest,output,plan['city'])
    retire.verified_bundle(output/'cache_manifest.json',output/'verification.json',canonical)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','output','protection','protected-sources','inventory','targets'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--limit',type=int,default=1000);a=p.parse_args()
    if a.limit<1:p.error('positive batch limit required')
    parent_plan=retire.read(a.parent/'plan.json')
    a.output.mkdir(parents=True,exist_ok=True)
    anchors=validation_anchors(parent_plan)
    protected_rows=retire.rows(a.protected_sources)
    for name in anchors:
        path=Path(name);st=path.stat()
        protected_rows.append(dict(path=str(path.absolute()),resolved_path=str(path.resolve()),dev=st.st_dev,ino=st.st_ino,reason='parent_final_raw_required_batch_validation'))
    protected_sources=a.output/'protected_sources_with_anchors.jsonl'
    protected_sources.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in protected_rows))
    retire.atomic_json(a.output/'validation_anchors.json',dict(paths=anchors,parent_plan_identity=parent_plan['identity']))
    live=retire.snapshot_live();protection=retire.compile_protection(retire.protection_snapshot(a.protection,protected_sources,live))
    selected,summary=select_entries(a.parent,parent_plan,protection,limit=a.limit,local_device=ROOT.stat().st_dev)
    a.output.mkdir(parents=True,exist_ok=True);retire.atomic_json(a.output/'selection.json',summary)
    print(json.dumps(summary),flush=True)
    publish_batch(a.parent,[r for _,r,_ in selected],a.output)
    plan=retire.create_plan(a.output/'cache_manifest.json',a.output/'verification.json',a.output/'canonical.jsonl',
                            a.inventory,a.targets,a.protection,protected_sources,a.output/'retirement')
    print(json.dumps(dict(plan=str(a.output/'retirement/deletion_plan.json'),eligible=sum(r['status']=='retire' for r in plan['rows']))),flush=True)

if __name__=='__main__':main()
