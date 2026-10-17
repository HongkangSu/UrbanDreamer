"""Finite-sample probability-score diagnostic; never selects or trains models."""
from pathlib import Path
import sys,json
import numpy as np
root=Path.cwd();study=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else root/'outputs/c2_convergence_20261015/v1';source=study/'source'
assert json.loads((study/'validation_selection.json').read_text())['status']=='frozen_before_new_blindtest'
assert json.loads((study/'summary.json').read_text())['status']=='complete'
sys.path.insert(0,str(source))
from tools.evaluate_pretrained_distribution import supplementary_scores
from tools.summarize_component_predictions import demand_bootstrap
from tools.component_experiment_data import sha256
rows=[];items={i['condition_index']:i for i in json.loads((study/'split.json').read_text())['test_conditions']}
variants=['B_parent','C2_epoch4','C2_converged'];metrics=['count_crps_u_all','count_crps_u_occupied','speed_crps_u_all','speed_crps_u_occupied']
for entry in json.loads((study/'test/bank_manifest.json').read_text()):
 assert sha256(Path(entry['path']))==entry['sha256'];item=items[entry['condition_index']]
 with np.load(entry['path']) as bank:
  for arm in variants:
   values=supplementary_scores(bank[arm],bank['target'],bank['scales'])
   rows.append(dict(condition_index=item['condition_index'],demand_group=item['demand_group'],episode=str(item['source_path'])+'::'+str(item['episode']),arm=arm,**{m:values[m] for m in metrics}))
summary={arm:{m:demand_bootstrap([r for r in rows if r['arm']==arm],m,repetitions=2000,seed=20261021) for m in metrics} for arm in variants};pairs={}
for label,a,b in [('extended_vs_B','C2_converged','B_parent'),('extended_vs_epoch4','C2_converged','C2_epoch4')]:
 left={r['condition_index']:r for r in rows if r['arm']==a};right={r['condition_index']:r for r in rows if r['arm']==b};diff=[]
 for index,r in left.items():
  other=right[index];diff.append(dict(demand_group=r['demand_group'],episode=r['episode'],**{m:None if r[m] is None else r[m]-other[m] for m in metrics}))
 pairs[label]={m:demand_bootstrap(diff,m,repetitions=2000,seed=20261021) for m in metrics}
result=dict(status='complete_diagnostic',formula='mean|X-y| - sum(i!=j)|Xi-Xj|/(2K(K-1));K8',used_for_selection=False,training=False,summary=summary,pairs=pairs,rows=rows,runner_sha256=sha256(Path(__file__)))
(study/'fair_crps_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n');print('fair_crps_diagnostic.json')
