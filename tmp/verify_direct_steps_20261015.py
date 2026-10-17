from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import verify_direct_context_fm_20261014 as v
study=Path('outputs/direct_context_fm_20261015/steps100_v1').resolve()
p=v.read(study/'protocol.json');s=v.read(study/'summary.json');items=v.read(study/'conditions.json')
assert s['status']=='complete' and not s['training'] and not s['selection']
assert p['steps']==[4,100] and not p['training'] and not p['selection']
launch=v.read(study/'launch.json');args=launch['argv'];parent=Path(args[args.index('--study')+1]);oldlaunch=v.read(parent/'launch.json')['argv'];options={oldlaunch[i]:oldlaunch[i+1] for i in range(len(oldlaunch)-1) if oldlaunch[i].startswith('--')}
refs=dict(checkpoint=Path(options['--experiment'])/'snapshot/checkpoints/latest.pt',split=parent/'split.json',selection=parent/'validation_selection.json',normalizers=parent/'normalizers.json',basis=Path(options['--basis']),weak_head=Path(options['--weak-head']),scale_reference=Path(options['--scale-reference']),parent_protocol=parent/'protocol.json',parent_source=parent/'source_manifest.json',cpu4_banks=parent/'test/bank_manifest.json')
refs.update({a:parent/'checkpoints'/f'best_{a}.pt' for a in p['arms']})
for k,path in refs.items():assert v.sha(path)==p['reference_hashes'][k]
for name,h in v.read(study/'source_manifest.json').items():assert v.sha(study/'source'/name)==h
assert all(Path(path).is_relative_to(study/'source') for path in v.read(study/'import_audit.json').values())
assert p['reference_hashes']['checkpoint']==v.CP_SHA
assert items==v.read(parent/'split.json')['test_conditions']
selection=v.read(refs['selection']);assert p['selected_epochs']=={a:selection['best'][a]['epoch'] for a in p['arms']}
rows=v.read(study/'states.json');banks=v.read(study/'bank_manifest.json');assert len(rows)==192
for steps in (4,100):
 arms=[f'{a}_s{steps}' for a in p['arms']]
 v.verify_banks(items,[r for r in rows if r['steps']==steps],banks,arms,8,steps)
seed=v.read(parent/'protocol.json')['seed'];v.verify_summary(rows,s['test'],seed)
pairs={}
for steps in (4,100):
 for label,a,b in [('soft_vs_zero','soft_direct','zero_direct'),('hint_vs_zero','condition_only','zero_direct'),('shift_vs_hint','soft_direct','condition_only')]:pairs[f'{label}_s{steps}']=(f'{a}_s{steps}',f'{b}_s{steps}')
for a in p['arms']:pairs[f'{a}_100minus4']=(f'{a}_s100',f'{a}_s4')
assert pairs.keys()==s['pairs'].keys()
for label,(a,b) in pairs.items():
 left={r['condition_index']:r for r in rows if r['arm']==a};right={r['condition_index']:r for r in rows if r['arm']==b}
 for metric,report in s['pairs'][label].items():
  diff=[]
  for index,r in left.items():
   o=right[index];assert (r[metric] is None)==(o[metric] is None)
   diff.append(dict(demand_group=r['demand_group'],episode=r['episode'],delta=None if r[metric] is None else r[metric]-o[metric]))
  v.interval(v.clusters(diff,'delta'),report,seed)
oldbanks={b['condition_index']:b for b in v.read(refs['cpu4_banks'])};reported={(r['condition_index'],r['arm']):r for r in v.read(study/'cpu4_backend_differences.json')}
for b in banks:
 old=oldbanks[b['condition_index']];assert v.sha(old['path'])==old['sha256']
 with np.load(b['path']) as actual,np.load(old['path']) as previous:
  for a in p['arms']:
   delta=actual[f'{a}_s4'].astype(np.float64)-previous[a]
   r=reported[(b['condition_index'],a)];v.close(r['max_absolute_difference'],np.abs(delta).max());v.close(r['rms_difference'],np.sqrt(np.square(delta).mean()))
assert len(v.read(study/'initial_gates.json'))==3 and all(g['zero_residual_exact'] for g in v.read(study/'initial_gates.json'))
result=dict(status='pass',fixed_selected_heads=True,same_test_noise_conditions=True,independent_numpy_metrics_and_CI=True,test_rows=len(rows),test_groups=16,integration_steps=[4,100],no_training_or_reselection=True,verifier_sha256=v.sha(__file__),**v.COUNTERS)
(study/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
