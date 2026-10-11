"""Matched frozen-WM imagination experiment; outputs are experimental, not strict resumes."""
from __future__ import annotations
import argparse,gc,hashlib,json,math,os,random,sys,time
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_online_pilot import atomic_json,isolated_evaluation,sha256_file
from tools.train_codec_ready import SparseGraphCodec
from urban_dreamer.learning.training import MainlineTrainer
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.config import load_config
from urban_dreamer.traffic.demand import demand_identity

def model_hash(model):
 digest=hashlib.sha256()
 for key,value in model.state_dict().items():
  digest.update(key.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
 return digest.hexdigest()

def tree_hash(value):
 digest=hashlib.sha256()
 def visit(item):
  if isinstance(item,torch.Tensor):
   item=item.detach().cpu().contiguous().numpy()
  if isinstance(item,np.ndarray):
   digest.update(str(item.dtype).encode());digest.update(str(item.shape).encode());digest.update(item.tobytes())
  elif isinstance(item,dict):
   for key in sorted(item,key=repr):visit(key);visit(item[key])
  elif isinstance(item,(tuple,list)):
   digest.update(str(type(item)).encode())
   for child in item:visit(child)
  else:digest.update(repr(item).encode())
  digest.update(b'\0')
 visit(value)
 return digest.hexdigest()

def distribution(actor,corpus):
 actor_device=next(actor.parameters()).device
 with torch.no_grad():
  probabilities=[]
  for index in range(0,len(corpus['features']),512):
   features=torch.from_numpy(corpus['features'][index:index+512]).to(actor_device)
   mask=torch.from_numpy(corpus['mask'][index:index+512]).to(actor_device)
   probabilities.append(actor.forward_logits(features,mask).softmax(-1).cpu())
  p=torch.cat(probabilities);mask=torch.from_numpy(corpus['mask']);cost=torch.expm1(torch.from_numpy(corpus['features'][...,4])*math.log1p(3600)).masked_fill(~mask,0)
  best=cost.masked_fill(~mask,torch.inf).amin(-1);greedy=p.argmax(-1)
  return dict(states=len(p),entropy=float(-(p*p.clamp_min(1e-30).log()).sum(-1).mean()),max_probability=float(p.amax(-1).mean()),shortest_action_mass=float((p*((cost-best[:,None]).abs()<1e-3)).sum(-1).mean()),nonprogress_mass=float((p*torch.from_numpy(corpus['nonprogress'])).sum(-1).mean()),expected_detour_seconds=float(((p*cost).sum(-1)-best).mean()),argmax_detour_seconds=float((cost.gather(1,greedy[:,None]).squeeze(1)-best).mean()))

def evaluate(trainer,directory,indices=(0,1,2)):
 result_path=directory/'results.json'
 rows=json.loads(result_path.read_text()) if result_path.exists() else []
 if [row['index'] for row in rows]!=list(indices[:len(rows)]):raise ValueError('cached evaluation membership differs')
 for index in indices[len(rows):]:
  with isolated_evaluation(trainer,directory):
   trainer.config=replace(trainer.config,run=replace(trainer.config.run,seed=314159))
   trainer._next_environment_episode=index
   random.seed(271828+index);np.random.seed(271828+index);torch.manual_seed(271828+index)
   started=time.monotonic();episode=trainer._run_episode(mode='online',episode_index=index)
   vehicles=trainer.demand.sample(seed=314159,episode_index=index)
   rows.append(dict(index=index,demand_hash=demand_identity(vehicles),population=episode.total_vehicle_count,metrics=episode.metrics,wall_seconds=time.monotonic()-started))
  atomic_json(directory/'results.json',rows)
 return rows

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--passes',type=int,default=64)
 parser.add_argument('--arms',nargs='+',default=['legacy','population'])
 parser.add_argument('--resume-from',type=Path)
 parser.add_argument('--evaluation-indices',nargs='+',type=int,default=[0,1,2])
 parser.add_argument('--evaluate-only',action='store_true')
 parser.add_argument('--output',type=Path,default=ROOT/'outputs/convergence_diagnosis_20261005/entropy')
 args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 if args.resume_from and args.evaluation_indices==[0,1,2]:
  original_before=json.loads((args.resume_from/'before/results.json').read_text())
  if len(original_before)!=3:raise ValueError('extension requires complete original baseline')
  atomic_json(args.output/'before/results.json',original_before)
 if 'URB_ACTOR_ENTROPY_MODE' not in (ROOT/'urban_dreamer/learning/training.py').read_text():raise RuntimeError('entropy mode hook is not installed yet')
 os.environ['URB_IMAGINATION_WORKERS']='8';torch.set_num_threads(4);torch.cuda.set_device(1);torch.cuda.set_per_process_memory_fraction(.06,1)
 source=ROOT/'outputs/convergence_diagnosis_20261005/frozen';path=source/'checkpoints/latest.pt'
 checkpoint=torch.load(path,map_location='cpu',weights_only=False)
 cfg=load_config(source/'run_config.toml');prepared=validate_prepared_inputs(cfg);topology=load_traffic_topology(prepared)
 with np.load(ROOT/'outputs/convergence_diagnosis_20261005/policy/fixed_observations.npz') as archive:corpus={key:archive[key] for key in archive.files}
 protocol=dict(evaluation_indices=args.evaluation_indices,evaluate_only=args.evaluate_only,resume_from=str(args.resume_from) if args.resume_from else None,source=str(path),source_sha256=sha256_file(path),source_online_episodes=checkpoint['next_unit'],passes=args.passes,arms=args.arms,rollouts_per_pass=8,max_windows=8,world_model_updates=0,stochastic_demand_seed=314159,evaluation_actor_sampling_seed=271828,experimental_override=True,note='Same initial full checkpoint and RNG each arm. WM frozen; V/Q/actor update normally. Three fixed stochastic SUMO demands. This is not a production strict continuation.')
 atomic_json(args.output/'protocol.json',protocol)
 for arm in args.arms:
  os.environ['URB_ACTOR_ENTROPY_MODE']=arm;directory=args.output/arm;directory.mkdir(parents=True,exist_ok=True)
  trainer=MainlineTrainer(config=replace(cfg,run=replace(cfg.run,output_dir=directory,device='cuda:1')),prepared_inputs=prepared,topology=topology)
  trainer.world_model.flow_projector.__class__=SparseGraphCodec
  trainer.world_model.flow_projector.register_buffer('sparse_adjacency',trainer._tensor(topology.network.adjacency).to_sparse().coalesce(),persistent=False)
  # Reload immutable source per arm: optimizer CPU step tensors may otherwise alias.
  arm_source=(args.resume_from/arm/'experimental_state.pt') if args.resume_from else path
  arm_checkpoint=torch.load(arm_source,map_location='cpu',weights_only=False)
  first_pass=arm_checkpoint.get('passes_completed',0)
  trainer.load_state_dict(arm_checkpoint['training_state']);del arm_checkpoint
  trainer.world_model.requires_grad_(False)
  initial_state=trainer.state_dict()
  atomic_json(directory/'initial_control.json',dict(source_sha256=sha256_file(arm_source),passes_completed=first_pass,hashes={key:tree_hash(initial_state[key]) for key in ('models','optimizers','rng')}))
  del initial_state
  initial_hash=model_hash(trainer.world_model);rows=[]
  try:
   before_path=args.output/'before/results.json'
   if not before_path.exists() or len(json.loads(before_path.read_text())) != len(args.evaluation_indices):
    if args.evaluate_only:
     trainer.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['training_state'])
     evaluate(trainer,args.output/'before',args.evaluation_indices)
     trainer.load_state_dict(torch.load(arm_source,map_location='cpu',weights_only=False)['training_state'])
     trainer.world_model.requires_grad_(False)
    else:evaluate(trainer,args.output/'before',args.evaluation_indices)
   rows.append(dict(pass_index=first_pass,distribution=distribution(trainer.actor,corpus)))
   for pass_index in range(first_pass+1,(first_pass if args.evaluate_only else args.passes)+1):
    started=time.monotonic();metrics=trainer._train_imagination_pass()
    coefficient=metrics.get('entropy_coefficient_mean')
    if coefficient is None:raise RuntimeError('training did not report entropy coefficient')
    if arm=='legacy' and not math.isclose(coefficient,.002,rel_tol=1e-4):raise RuntimeError('legacy coefficient mismatch')
    if arm=='population' and not 0<coefficient<.0001:raise RuntimeError('population coefficient mismatch')
    if arm=='off' and coefficient!=0:raise RuntimeError('off coefficient mismatch')
    row=dict(pass_index=pass_index,metrics=metrics,wall_seconds=time.monotonic()-started)
    if pass_index%16==0:row['distribution']=distribution(trainer.actor,corpus)
    rows.append(row)
    with (directory/'passes.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
    print(json.dumps(dict(arm=arm,**row)),flush=True)
   final_hash=model_hash(trainer.world_model)
   if initial_hash!=final_hash:raise RuntimeError('World model changed during isolated ablation')
   saved=dict(experimental_protocol=protocol,arm=arm,passes_completed=args.passes,training_state=trainer.state_dict());torch.save(saved,directory/'experimental_state.pt')
   evaluation=evaluate(trainer,directory/'after',args.evaluation_indices)
   before=json.loads((args.output/'before/results.json').read_text())
   if [(r['index'],r['demand_hash'],r['population']) for r in before]!=[(r['index'],r['demand_hash'],r['population']) for r in evaluation]:raise RuntimeError('paired demand identity differs')
   atomic_json(directory/'result.json',dict(arm=arm,passes=args.passes,world_model_hash=final_hash,world_model_unchanged=True,training=rows,after=evaluation))
  finally:trainer.close();del trainer;gc.collect();torch.cuda.empty_cache()
 atomic_json(args.output/'status.json',dict(status='complete',arms=args.arms,passes=args.passes))
if __name__=='__main__':main()
