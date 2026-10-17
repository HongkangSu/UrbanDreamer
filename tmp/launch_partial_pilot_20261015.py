from pathlib import Path
import json,hashlib,shutil,os,subprocess
root=Path.cwd();parent=root/'outputs/direct_context_fm_20261015/steps100_v1';out=root/'outputs/partial_context_fm_20261015/pilot_v1';split=root/'outputs/partial_context_fm_20261015/v1/split.json'
assert not out.exists();out.mkdir(parents=True)
source=out/'source';shutil.copytree(parent/'source',source)
manifest=json.loads((parent/'source_manifest.json').read_text());h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name in ['tools/train_partial_context_fm.py','tests/test_partial_context_fm.py']:
 shutil.copy2(root/name,source/name);manifest[name]=h(source/name)
for name,expected in manifest.items():assert h(source/name)==expected
(out/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
cmd=['/home/wangwenrui/opt/miniconda3/envs/URB/bin/python','-u',str(source/'tools/train_partial_context_fm.py'),'--experiment',str(root/'outputs/component_experiments_20261011'),'--split',str(split),'--basis',str(root/'outputs/pretrained_prior_rank_20261013/v1/exact_basis.npz'),'--weak-head',str(root/'outputs/soft_prior_fm_20261014/matched_hybrid_head/head.npz'),'--output',str(out),'--pilot-only','--device','cuda:1','--gpu-memory-fraction','.08','--batch-graphs','2']
env=dict(os.environ,PARTIAL_FM_FROZEN_ROOT=str(source),DIRECT_FM_FROZEN_ROOT=str(source),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(source))
with (out/'run.log').open('w') as log:p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'launch.json').write_text(json.dumps(dict(pid=p.pid,argv=cmd,runner_sha256=h(source/'tools/train_partial_context_fm.py')),indent=2)+'\n');print(json.dumps(dict(pid=p.pid,output=str(out))))
