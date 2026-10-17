from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
import verify_direct_context_fm_20261014 as v
study=Path(sys.argv[1]).resolve();main=v.read(study/'summary.json');protocol=v.read(study/'h4_followup_protocol.json');gate=v.read(study/'h4_gate.json');eligible=[]
for arm,label in protocol['eligible_pairs'].items():
 p=main['pairs'][label]
 if all(p[m]['ci_high'] is not None and p[m]['ci_high']<0 for m in ('energy_score','count_crps_occupied','speed_crps_occupied')) and p['total_count_absolute_error']['estimate']<=0:eligible.append(arm)
assert eligible==gate['eligible'] and not gate['tuning_after_test']
if eligible:
 out=study/'h4_followup';rows=v.read(out/'states.json');s=v.read(out/'summary.json');items=v.read(study/'split.json')['test_conditions'][:8];p=v.read(out/'protocol.json');assert not p['training']
 seed=v.read(study/'protocol.json')['seed'];v.verify_banks(items,rows,v.read(out/'bank_manifest.json'),['B']+eligible,8,100,horizon=4);v.verify_summary(rows,s['test'],seed)
 for arm,metrics in s['pairs'].items():
  left={r['condition_index']:r for r in rows if r['arm']==arm};right={r['condition_index']:r for r in rows if r['arm']=='B'}
  for metric,report in metrics.items():
   diff=[dict(demand_group=r['demand_group'],episode=r['episode'],delta=None if r[metric] is None else r[metric]-right[i][metric]) for i,r in left.items()]
   v.interval(v.clusters(diff,'delta'),report,seed)
 for arm,h in p['checkpoint_hashes'].items():assert v.sha(study/'checkpoints'/f'best_{arm}.pt')==h
 result=dict(status='pass',eligible=eligible,independent_numpy_metrics_and_CI=True,**v.COUNTERS)
 (out/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
else:
 assert not (study/'h4_followup').exists();result=dict(status='gate_correct_no_followup',eligible=[])
print(json.dumps(result))
