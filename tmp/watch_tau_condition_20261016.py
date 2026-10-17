import json,subprocess,time
from pathlib import Path
root=Path('/home/wangwenrui/shk/UrbanDreamer/urban_dreamer');output=root/'outputs/tau_condition_20261016/v1';pid=1817439
while not (output/'summary.json').exists():
    if not Path(f'/proc/{pid}').exists():raise SystemExit('tau prediction process exited before summary')
    time.sleep(15)
subprocess.run(['/home/wangwenrui/opt/miniconda3/envs/URB/bin/python',str(root/'tools/report_tau_condition.py')],cwd=root,check=True)
print('tau sensitivity verification completed',flush=True)
