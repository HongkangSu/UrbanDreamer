import json,hashlib,shutil,subprocess,os
from pathlib import Path
root=Path.cwd();old=root/'outputs/direct_context_fm_20261014/v1';out=root/'outputs/direct_context_fm_20261014/v2_cpu4'
assert not out.exists();out.mkdir()
for name in ['cache_manifest.json','split.json','data_split_audit.json']:shutil.copy2(old/name,out/name)
split=json.loads((out/'split.json').read_text());split['cache_manifest_path']=str(out/'cache_manifest.json');(out/'split.json').write_text(json.dumps(split,indent=2)+'\n')
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
audit=json.loads((out/'data_split_audit.json').read_text());audit['split_sha256']=h(out/'split.json');(out/'data_split_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
source=out/'source';shutil.copytree(old/'source',source)
shutil.copy2(root/'tools/train_direct_context_fm.py',source/'tools/train_direct_context_fm.py');shutil.copy2(root/'tests/test_direct_context_fm.py',source/'tests/test_direct_context_fm.py')
manifest=json.loads((old/'source_manifest.json').read_text())
for name in manifest:manifest[name]=h(source/name)
(out/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
refs=json.loads((root/'tmp/direct_context_cli_refs_20261014.json').read_text())
cmd=['/home/wangwenrui/opt/miniconda3/envs/URB/bin/python','-u',str(source/'tools/train_direct_context_fm.py'),'--experiment',str(root/'outputs/component_experiments_20261011'),'--split',str(out/'split.json'),'--parent',str(root/'outputs/nonlinear_prior_fm_20261014/v1'),'--weak-head',str(root/'outputs/soft_prior_fm_20261014/matched_hybrid_head/head.npz'),'--basis',str(root/'outputs/pretrained_prior_rank_20261013/v1/exact_basis.npz'),'--known-reference',str(root/'outputs/soft_prior_fm_20261014/primary100_hybrid_v1'),'--known-cache',refs['known_cache'],'--scale-reference',refs['scales'],'--output',str(out),'--device','cpu','--threads','4','--steps','4']
env=dict(os.environ,DIRECT_FM_FROZEN_ROOT=str(source),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4',PYTHONPATH=str(source))
with (out/'run.log').open('w') as log:p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=root)
(out/'launch.json').write_text(json.dumps(dict(pid=p.pid,argv=cmd,source_files=len(manifest),runner_sha256=h(source/'tools/train_direct_context_fm.py'),reason='GPU admission v1 failed before scores; CPU4 Euler4 screening specified before anyfit/scores',original_v1_untouched=True),indent=2)+'\n')
print(json.dumps(dict(pid=p.pid,output=str(out))))
