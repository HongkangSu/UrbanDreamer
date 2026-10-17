from pathlib import Path
import json,hashlib,shutil,os,subprocess
root=Path.cwd();parent=root/'outputs/partial_context_fm_20261015/v2_gpu0';out=root/'outputs/c2_convergence_20261015/v1';source=out/'source'
assert not source.exists();assert not (out/'training_history.json').exists()
shutil.copytree(parent/'source',source);manifest=json.loads((parent/'source_manifest.json').read_text());h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name in ['tools/continue_partial_context_fm.py','tests/test_continue_partial_context_fm.py']:
 shutil.copy2(root/name,source/name);manifest[name]=h(source/name)
for name,expected in manifest.items():assert h(source/name)==expected
(out/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
cmd=['/home/wangwenrui/opt/miniconda3/envs/URB/bin/python','-u',str(source/'tools/continue_partial_context_fm.py'),'--parent',str(parent),'--split',str(out/'split.json'),'--output',str(out),'--device','cuda:0','--gpu-memory-fraction','.08','--maximum-epoch','80']
env=dict(os.environ,PARTIAL_FM_FROZEN_ROOT=str(source),DIRECT_FM_FROZEN_ROOT=str(source),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(source))
with (out/'run.log').open('w') as log:p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'launch.json').write_text(json.dumps(dict(pid=p.pid,argv=cmd,runner_sha256=h(source/'tools/continue_partial_context_fm.py'),source_files=len(manifest),design_sha256=h(out/'design.json')),indent=2)+'\n')
print(json.dumps(dict(pid=p.pid,output=str(out))))
