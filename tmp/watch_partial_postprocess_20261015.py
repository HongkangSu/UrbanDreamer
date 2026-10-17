from pathlib import Path
import subprocess,sys,time,os,json
root=Path.cwd();study=root/'outputs/partial_context_fm_20261015/v2_gpu0';pid=996234
while not (study/'summary.json').exists():
 try:os.kill(pid,0)
 except ProcessLookupError:raise SystemExit('Training ended without summary; inspect run.log')
 time.sleep(10)
commands=[['tmp/verify_partial_context_fm_20261015.py','--study',str(study)],['tmp/evaluate_partial_h4_20261015.py',str(study)],['tmp/verify_partial_h4_20261015.py',str(study)],['tmp/report_partial_context_fm_20261015.py']]
for args in commands:
 print('RUN '+args[0],flush=True);subprocess.run([sys.executable,*args],check=True)
(study/'postprocess_complete.json').write_text(json.dumps(dict(status='complete',steps=[a[0] for a in commands]),indent=2)+'\n')
print('COMPLETE',flush=True)
