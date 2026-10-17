from pathlib import Path
import json,hashlib,shutil,os,subprocess
root=Path.cwd();pilot=root/'outputs/partial_context_fm_20261015/pilot_v1';out=root/'outputs/partial_context_fm_20261015/v1';source=out/'source'
assert not source.exists();assert not (out/'training_history.json').exists()
shutil.copytree(pilot/'source',source)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((pilot/'source_manifest.json').read_text())
for name in ['tools/train_partial_context_fm.py','tests/test_partial_context_fm.py','tools/run_partial_context_fm.py','tests/test_run_partial_context_fm.py']:
 shutil.copy2(root/name,source/name);manifest[name]=h(source/name)
for name,expected in manifest.items():assert h(source/name)==expected
(out/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
cmd=['/home/wangwenrui/opt/miniconda3/envs/URB/bin/python','-u',str(source/'tools/run_partial_context_fm.py'),'--experiment',str(root/'outputs/component_experiments_20261011'),'--split',str(out/'split.json'),'--pilot',str(pilot),'--basis',str(root/'outputs/pretrained_prior_rank_20261013/v1/exact_basis.npz'),'--weak-head',str(root/'outputs/soft_prior_fm_20261014/matched_hybrid_head/head.npz'),'--scale-reference',str(root/'outputs/pretrained_evaluation_20261012/distribution_full_v1/evaluation_8groups/protocol.json'),'--output',str(out),'--minimum-epochs','3','--maximum-epochs','4','--validation-every','1','--batch-graphs','2','--device','cuda:1','--gpu-memory-fraction','.08']
env=dict(os.environ,PARTIAL_FM_FROZEN_ROOT=str(source),DIRECT_FM_FROZEN_ROOT=str(source),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(source))
with (out/'run.log').open('w') as log:p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'launch.json').write_text(json.dumps(dict(pid=p.pid,argv=cmd,runner_sha256=h(source/'tools/run_partial_context_fm.py'),budget_basis='Training-only pilot tripleepoch348sec; cap4equalepochs~25min plusvalidation; chosen BEFORE anypartialvalidation/testscore',source_files=len(manifest),initial_data_manifest_sha256=h(out/'cache_manifest.json')),indent=2)+'\n')
print(json.dumps(dict(pid=p.pid,output=str(out))))
