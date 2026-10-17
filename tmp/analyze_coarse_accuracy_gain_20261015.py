from pathlib import Path
import os,sys,json
root=Path.cwd();parent=root/'outputs/partial_context_fm_20261015/v2_gpu0';source=parent/'source'
os.environ['PARTIAL_FM_FROZEN_ROOT']=str(source);os.environ['DIRECT_FM_FROZEN_ROOT']=str(source);sys.path.insert(0,str(source))
import numpy as np
import torch
from scipy.stats import rankdata
from tools.train_partial_context_fm import load_frozen,MeanOverridePrior,frozen_window_batch,CompactPretrainDataset
from tools.component_experiment_data import sha256
read=lambda p:json.loads(p.read_text());torch.set_num_threads(2);torch.set_num_interop_threads(2)
args=read(parent/'launch.json')['argv'];opts={args[i]:args[i+1] for i in range(len(args)-1) if args[i].startswith('--')}
config,prepared,topology,model,_=load_frozen(Path(opts['--experiment']),device='cpu');model.eval().requires_grad_(False)
with np.load(opts['--basis']) as a:basis,eigen=a['basis'],a['eigenvalues']
with np.load(opts['--weak-head']) as a:coef=a['coefficients']
model.dynamics.graph_prior=MeanOverridePrior(model.dynamics.graph_prior,operator=(basis*(1/(1+.1*eigen)))@basis.T,coefficients=coef,state_only=True)
split=read(parent/'split.json');data=CompactPretrainDataset(config,prepared,topology,1,cache_manifest_path=Path(split['cache_manifest_path']),split='validation');capacities=torch.tensor(topology.network.capacities,dtype=torch.float32).clamp_min(1)
old=read(parent/'test/states.json');index={(r['condition_index'],r['arm']):r for r in old};rows=[]
for item in split['test_conditions']:
 batch=data.batch_from_starts(item['episode_index'],[item['start']],materialization_horizon=1,fm_only=True)
 x=frozen_window_batch(model,batch,capacities);delta=x['target']-x['z'];residual=delta-x['b'];zero=float(delta.double().square().sum());coarse=float(residual.double().square().sum());i=item['condition_index'];b=index[(i,'B')];c=index[(i,'C2')]
 rows.append(dict(condition_index=i,demand_group=item['demand_group'],zero_increment_sse=zero,coarse_sse=coarse,coarse_skill=1-coarse/max(zero,1e-12),ES_gain_B_minus_C2=b['energy_score']-c['energy_score'],B_ES=b['energy_score'],C2_ES=c['energy_score']))
order=sorted(rows,key=lambda r:r['coarse_skill']);quartiles=[]
for q in range(4):
 selected=order[q*8:(q+1)*8];quartiles.append(dict(quartile=q+1,n=8,mean_coarse_skill=float(np.mean([r['coarse_skill'] for r in selected])),mean_ES_gain=float(np.mean([r['ES_gain_B_minus_C2'] for r in selected])),C2_wins=sum(r['ES_gain_B_minus_C2']>0 for r in selected)))
x=np.array([r['coarse_skill'] for r in rows]);y=np.array([r['ES_gain_B_minus_C2'] for r in rows]);rho=float(np.corrcoef(rankdata(x),rankdata(y))[0,1]);groups=sorted({r['demand_group'] for r in rows});rng=np.random.default_rng(20261015);boot=[]
for _ in range(2000):
 chosen=rng.choice(groups,len(groups),replace=True);selected=[r for g in chosen for r in rows if r['demand_group']==g];xx=rankdata([r['coarse_skill'] for r in selected]);yy=rankdata([r['ES_gain_B_minus_C2'] for r in selected])
 if np.std(xx)>0 and np.std(yy)>0:boot.append(float(np.corrcoef(xx,yy)[0,1]))
result=dict(status='exploratory_existing_parent_test_only',parent_state_hash=sha256(parent/'test/states.json'),aggregate_skill=1-sum(r['coarse_sse'] for r in rows)/sum(r['zero_increment_sse'] for r in rows),spearman_accuracy_vs_ES_gain=rho,cluster_bootstrap_ci=np.quantile(boot,[.025,.975]).tolist(),quartiles=quartiles,rows=rows,limitations='32oldstates16groups, correlationnotcausation; currentcontinuationtestuntouched; nottrainingorschedulerinput')
out=root/'outputs/c2_convergence_20261015/v1/parent_coarse_accuracy_analysis.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
# Previously verified larger train/validation source-alignment evidence.
a=read(root/'outputs/nonlinear_prior_fm_20261014/v1/alignment_stats.json')
for stage in ['train','validation']:
 selected=[r for r in a if r['stage']==stage];print(stage,dict(no_noise_skill=1-sum(r['weak_no_noise_sse'] for r in selected)/sum(r['zero_delta_energy'] for r in selected),paired_noisy_source_cost_reduction=1-sum(r['weak_noisy_sse'] for r in selected)/sum(r['zero_noisy_sse'] for r in selected)))
