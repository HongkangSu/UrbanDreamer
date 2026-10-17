"""Execute only the prespecified positive-result C0/H4 follow-up, without tuning."""
from pathlib import Path
import json,os,sys,time
root=Path.cwd();study=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else root/'outputs/partial_context_fm_20261015/v2_gpu0';source=study/'source';out=study/'h4_followup'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ['PARTIAL_FM_FROZEN_ROOT']=str(source);os.environ['DIRECT_FM_FROZEN_ROOT']=str(source)
sys.path.insert(0,str(source))
import numpy as np
import torch
from tools.run_partial_context_fm import (read,load_selected,ARMS,DirectResidual,partial_context,MeanOverridePrior,
 load_frozen,sha256,atomic_json,CompactPretrainDataset,trajectory_bank,endpoint_target,score,study_summary,paired_metrics)
from urban_dreamer.learning.cohort_targets import prepare_cohort_batch
s=read(study/'summary.json');p=read(study/'h4_followup_protocol.json');selection=read(study/'validation_selection.json');split=read(study/'split.json')
assert s['status']=='complete' and selection['status']=='frozen_before_new_blindtest'
eligible=[]
for arm,label in p['eligible_pairs'].items():
 pair=s['pairs'][label]
 if all(pair[m]['ci_high'] is not None and pair[m]['ci_high']<0 for m in ('energy_score','count_crps_occupied','speed_crps_occupied')) and pair['total_count_absolute_error']['estimate']<=0:eligible.append(arm)
atomic_json(study/'h4_gate.json',dict(eligible=eligible,protocol_sha256=sha256(study/'h4_followup_protocol.json'),tuning_after_test=False))
if not eligible:
 print(json.dumps(dict(status='gate_failed',eligible=[])));sys.exit(0)
assert not out.exists();out.mkdir();started=time.time();torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.08,device)
launch=read(study/'launch.json')['argv'];opts={launch[i]:launch[i+1] for i in range(len(launch)-1) if launch[i].startswith('--')}
config,prepared,topology,model,_=load_frozen(Path(opts['--experiment']),device=device);model.eval().requires_grad_(False);model.set_online_chunk_profile();native=model.dynamics.graph_prior
with np.load(opts['--basis']) as a:basis,eigenvalues=a['basis'],a['eigenvalues']
with np.load(opts['--weak-head']) as a:coefficients=a['coefficients']
model.dynamics.graph_prior=MeanOverridePrior(native,operator=(basis*(1/(1+.1*eigenvalues)))@basis.T,coefficients=coefficients,state_only=True)
capacities=torch.tensor(topology.network.capacities,dtype=torch.float32).clamp_min(1);limits=torch.tensor(topology.network.speed_limits_mps,dtype=torch.float32)
scales=np.asarray(read(Path(opts['--scale-reference']))['scales']['scales']);normalizers=read(study/'normalizers.json')['scales'];seed=int(opts.get('--seed',20261021))
data=CompactPretrainDataset(config,prepared,topology,5,cache_manifest_path=Path(split['cache_manifest_path']),split='validation')
heads={};states={};hashes={}
for arm in ['B']+eligible:
 heads[arm]=DirectResidual().to(device).eval().requires_grad_(False);path=study/'checkpoints'/f'best_{arm}.pt';hashes[arm]=sha256(path);states[arm]=torch.load(path,map_location=device,weights_only=True)['state']
atomic_json(out/'protocol.json',dict(parent_protocol=p,eligible=eligible,conditions=split['test_conditions'][:8],checkpoint_hashes=hashes,runner_sha256=sha256(Path(__file__)),training=False))
rows=[];banks=[]
with torch.no_grad():
 for ordinal,item in enumerate(split['test_conditions'][:8]):
  batch=data.batch_from_starts(item['episode_index'],[item['start']],fm_only=False);cohort=prepare_cohort_batch(batch,capacities=capacities)
  z=model.encode(cohort.transition_flow[:,0].to(device));truth=endpoint_target(cohort.transition_flow[:,4],batch.flow[:,5],capacities,limits).numpy();stored=dict(target=truth,scales=scales)
  for arm in ['B']+eligible:
   load_selected(model,heads[arm],states[arm])
   with partial_context(model.dynamics,heads[arm],arm,normalizers):
    draws=trajectory_bank(model,z,cohort.transition_action.to(device),cohort.transition_action_events.to(device),capacities=capacities.to(device),limits=limits.to(device),seed=item['condition_seed'],horizons=(4,),samples=8,chunk=4,steps=100)[4]
   stored[arm]=draws;rows.append(score(draws,truth,scales,int(cohort.cohort_count[0]),limits.numpy(),dict(item,arm=arm,episode=str(item['source_path'])+'::'+str(item['episode']),steps=100,horizon=4,k=8)))
  path=out/f'condition_{ordinal:03d}.npz';np.savez_compressed(path,**stored);banks.append(dict(condition_index=item['condition_index'],path=str(path.resolve()),sha256=sha256(path)))
  print(json.dumps(dict(conditions_complete=ordinal+1,total=8,elapsed_seconds=time.time()-started)),flush=True)
for arm,h in hashes.items():assert sha256(study/'checkpoints'/f'best_{arm}.pt')==h
for name,h in read(study/'source_manifest.json').items():assert sha256(source/name)==h
atomic_json(out/'states.json',rows);atomic_json(out/'bank_manifest.json',banks)
atomic_json(out/'summary.json',dict(status='complete',test=study_summary(rows,seed),pairs={a:paired_metrics(rows,a,'B',seed) for a in eligible},elapsed_seconds=time.time()-started))
