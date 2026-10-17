from pathlib import Path
import json,hashlib,shutil,os,subprocess
root=Path.cwd();parent=root/'outputs/direct_context_fm_20261014/v2_cpu4';out=root/'outputs/direct_context_fm_20261015/steps100_v1'
assert not out.exists();out.mkdir(parents=True)
source=out/'source';shutil.copytree(parent/'source',source)
shutil.copy2(root/'tools/evaluate_direct_context_steps.py',source/'tools/evaluate_direct_context_steps.py')
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((parent/'source_manifest.json').read_text());manifest['tools/evaluate_direct_context_steps.py']=h(source/'tools/evaluate_direct_context_steps.py')
for name,expected in manifest.items():assert h(source/name)==expected
(out/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
cmd=['/home/wangwenrui/opt/miniconda3/envs/URB/bin/python','-u',str(source/'tools/evaluate_direct_context_steps.py'),'--study',str(parent),'--output',str(out),'--device','cuda:1','--gpu-memory-fraction','.04']
env=dict(os.environ,DIRECT_FM_FROZEN_ROOT=str(source),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(source))
with (out/'run.log').open('w') as log:p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'launch.json').write_text(json.dumps(dict(pid=p.pid,argv=cmd,runner_sha256=h(source/'tools/evaluate_direct_context_steps.py'),parent_source_sha256=h(parent/'source_manifest.json')),indent=2)+'\n')
print(json.dumps(dict(pid=p.pid,output=str(out))))
