from pathlib import Path
import json,sys,time,subprocess,os
root=Path.cwd();study=root/'outputs/c2_convergence_20261015/v1';pid=1201579
while not (study/'summary.json').exists():
 try:os.kill(pid,0)
 except ProcessLookupError:raise SystemExit('Training exited without summary; inspect run.log')
 time.sleep(10)
for script in ['tmp/verify_c2_convergence_20261015.py','tmp/plot_c2_convergence_20261015.py']:
 args=['--study',str(study)] if 'verify_' in script else [str(study)]
 print('RUN '+script,flush=True);subprocess.run([sys.executable,script,*args],check=True)
status=json.loads((study/'summary.json').read_text())['status']
if status!='complete':
 print('BUDGET_LIMITED_TEST_CLOSED_NEEDS_RESUME',flush=True);sys.exit(0)
for script in ['tmp/score_c2_fair_crps_20261015.py','tmp/evaluate_c2_convergence_h4_20261015.py','tmp/verify_c2_convergence_h4_20261015.py','tmp/report_c2_convergence_20261015.py']:
 print('RUN '+script,flush=True);subprocess.run([sys.executable,script,str(study)],check=True)
(study/'postprocess_complete.json').write_text(json.dumps(dict(status='complete'),indent=2)+'\n');print('COMPLETE',flush=True)
