from pathlib import Path
import json,hashlib,shutil,os,subprocess
root=Path.cwd();parent=root/'outputs/partial_context_fm_20261015/v2_gpu0';out=root/'outputs/source_condition_controls_20261015/v1';source=out/'source'
assert not source.exists()
shutil.copytree(root/'outputs/c2_convergence_20261015/v1/source',source,ignore=shutil.ignore_patterns('__pycache__'))
manifest=json.loads((root/'outputs/c2_convergence_20261015/v1/source_manifest.json').read_text())
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name in ['tools/source_condition_controls.py','tools/train_source_condition_control.py','tests/test_source_condition_controls.py','tests/test_train_source_condition_control.py']:
 shutil.copy2(root/name,source/name);manifest[name]=h(source/name)
for name,expected in manifest.items():assert h(source/name)==expected
(out/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
design=dict(status='prespecified_before_training_or_validation',arms=dict(G=dict(source='N(0,I) dense iid',background_hint=False,reference='Z'),A=dict(source='Z+sigmaU16xi',background_hint=False,reference='Z'),B=dict(source='Z+sigmaU16xi',background_hint=True,reference='Z'),C2=dict(source='Z+b+sigmaU16xi',background_hint=True,reference='Z+b')),trainable_parameters=1414528,completed_pretrain_only=True,initialize='A/G completed pretrain plus identical zero-output g; B resume epoch4 including AdamW; C2 reuse validation-best epoch10 from completed plateau run',training_groups=64,validation_groups=16,validation_states=8,training_views=3072,batch_graphs=2,selection='same validation ES plateau rules, two LR reductions .3 patience5 final LR patience5',budget='independent convergence stopping; actual updates may differ; not strict equal-budget comparison',normalizers='train-only; G relative-Xt RMS re-fitted for unit full-rank source, A/B parent RMS',shared_test='fresh 32 states/16 groups; unopened until A/G/B ready; C2 fixed before this split',test_opened=False,reference_C2=dict(path=str((root/'outputs/c2_convergence_20261015/v1/checkpoints/best_C2.pt').resolve()),sha256=h(root/'outputs/c2_convergence_20261015/v1/checkpoints/best_C2.pt')),comparisons=['B minus A: background condition incremental value','A minus G: source distribution change (mean and covariance)','C2 minus B: source/reference usage beyond condition'],limitations=['Partial adaptation of frozen pretrained prefix; not full optimization of ordinary Gaussian FM','One training/adaptation seed, no policy benefit claim'])
(out/'design.json').write_text(json.dumps(design,indent=2)+'\n')
env=dict(os.environ,PARTIAL_FM_FROZEN_ROOT=str(source),DIRECT_FM_FROZEN_ROOT=str(source),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(source))
launches={}
for arm,device in [('B','cuda:0'),('A','cuda:1'),('G','cuda:1')]:
 (out/arm).mkdir()
 cmd=['/home/wangwenrui/opt/miniconda3/envs/URB/bin/python','-u',str(source/'tools/train_source_condition_control.py'),'--arm',arm,'--parent',str(parent),'--study',str(out),'--device',device,'--gpu-memory-fraction','.05','--maximum-epoch','80']
 with (out/arm/'run.log').open('w') as log:p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 launches[arm]=dict(pid=p.pid,argv=cmd)
(out/'launch.json').write_text(json.dumps(dict(arms=launches,source_files=len(manifest),source_manifest_sha256=h(out/'source_manifest.json'),design_sha256=h(out/'design.json')),indent=2)+'\n')
print(json.dumps(dict(arms=launches,output=str(out))))
