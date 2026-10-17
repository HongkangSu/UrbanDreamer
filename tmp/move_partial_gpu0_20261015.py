from pathlib import Path
import json,hashlib,os,signal,shutil,subprocess,datetime
root=Path.cwd();old=root/'outputs/partial_context_fm_20261015/v1';out=root/'outputs/partial_context_fm_20261015/v2_gpu0'
launch=json.loads((old/'launch.json').read_text());pid=launch['pid']
assert pid==977393 and not (old/'validation_selection.json').exists() and not (old/'test').exists()
command=Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\x00',' ')
assert str(old/'source/tools/run_partial_context_fm.py') in command
assert not out.exists();out.mkdir()
os.kill(pid,signal.SIGTERM)
reason=dict(status='resource_restart_before_first_epoch_complete',reason='GPU1 shared by two other compute jobs; GPU0 idle. Stop only owned training, restart unchangedprotocol/initialization onGPU0.',pid=pid,discarded_partial_training=True,new_test_scored=False,time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(old/'stop_reason.json').write_text(json.dumps(reason,indent=2)+'\n')
for name in ['split.json','cache_manifest.json','data_split_audit.json','h4_followup_protocol.json']:
 shutil.copy2(old/name,out/name)
shutil.copytree(old/'source',out/'source');shutil.copy2(old/'source_manifest.json',out/'source_manifest.json')
cmd=list(launch['argv']);cmd[2]=str(out/'source/tools/run_partial_context_fm.py');cmd[cmd.index('--split')+1]=str(out/'split.json');cmd[cmd.index('--output')+1]=str(out);cmd[cmd.index('--device')+1]='cuda:0'
source=out/'source';env=dict(os.environ,PARTIAL_FM_FROZEN_ROOT=str(source),DIRECT_FM_FROZEN_ROOT=str(source),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(source))
with (out/'run.log').open('w') as log:p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
new=dict(launch,pid=p.pid,argv=cmd,resource_restart_from=str(old),same_source_and_scientific_protocol=True,initial_data_manifest_sha256=hashlib.sha256((out/'cache_manifest.json').read_bytes()).hexdigest())
(out/'launch.json').write_text(json.dumps(new,indent=2)+'\n');print(json.dumps(dict(pid=p.pid,output=str(out))))
