"""Finish the prespecified pipeline after independent selection; no live code imports."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
study=Path(sys.argv[1]).resolve();post=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def atomic(path,value):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)
def gate():
 for name,h in read(post/'manifest.json').items():
  assert hashlib.sha256((post/name).read_bytes()).hexdigest()==h,name
launch=read(study/'launch.json')['arms'];python='/home/wangwenrui/opt/miniconda3/envs/URB/bin/python'
env=dict(os.environ,PARTIAL_FM_FROZEN_ROOT=str(study/'source'),DIRECT_FM_FROZEN_ROOT=str(study/'source'),PYTHONPATH=str(study/'source'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
try:
 while True:
  statuses={}
  for arm,item in launch.items():
   summary=study/arm/'summary.json'
   if summary.exists():
    value=read(summary);statuses[arm]=value['status']
    if value['status']!='ready_for_shared_test':raise RuntimeError(arm+' ended without convergence; test remains closed')
   else:
    path=Path('/proc')/str(item['pid'])/'stat'
    if not path.exists() or path.read_text().split(')')[1].split()[0]=='Z':raise RuntimeError(arm+' trainer exited before a summary; test remains closed')
    progress=study/arm/'progress.json'
    statuses[arm]=dict(status='training',epoch=read(progress)['epoch'] if progress.exists() else 0)
  atomic(study/'pipeline_progress.json',dict(stage='waiting_for_independent_validation_selection',arms=statuses,test_opened=False))
  if all(s=='ready_for_shared_test' for s in statuses.values()):break
  time.sleep(30)
 gate()
 stages=[('independent_training_audit',[python,str(post/'verify_source_condition_training_20261015.py'),'--study',str(study)]),
         ('shared_blind_test',[python,str(post/'evaluate_source_condition_controls.py'),'--study',str(study),'--device','cuda:0','--gpu-memory-fraction','.05']),
         ('independent_test_audit',[python,str(post/'verify_source_condition_test_20261015.py'),'--study',str(study)])]
 for name,cmd in stages:
  gate();atomic(study/'pipeline_progress.json',dict(stage=name,test_opened=name in ('shared_blind_test','independent_test_audit')))
  with (study/(name+'.log')).open('w') as log:
   subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 gate();atomic(study/'pipeline_progress.json',dict(stage='complete',independent_training_and_test_audits=True,test_opened=True))
except Exception as error:
 atomic(study/'pipeline_progress.json',dict(stage='failed',error=str(error),test_outputs_exist=(study/'shared_test').exists()))
 raise
