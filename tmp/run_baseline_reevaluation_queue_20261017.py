import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=Path((ROOT/'tmp/full_reevaluation_latest_path.txt').read_text().strip());M=OUT/'manifest.json';entries=json.loads(M.read_text())['entries'];start=time.time()
def run(e):
 out=Path(e['evaluation_output']);result=out/'result.json'
 pid_file=out/'evaluation.pid'
 if pid_file.exists():
  pid=int(pid_file.read_text())
  while True:
   try:
    argv=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    alive=any(x.endswith(b'/reevaluate_baseline_metrics_20261017.py') for x in argv) and e['city'].encode() in argv and e['method'].encode() in argv
   except OSError:alive=False
   if not alive:break
   time.sleep(2)
 if result.exists() and json.loads(result.read_text()).get('status')=='completed':return dict(city=e['city'],method=e['method'],exit_code=0,cached=True)
 cmd=[sys.executable,'-u',str(ROOT/'tmp/reevaluate_baseline_metrics_20261017.py'),'--manifest',str(M),'--city',e['city'],'--method',e['method']]
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 with (out/'evaluation.log').open('a') as log:
  proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env);(out/'evaluation.pid').write_text(str(proc.pid));code=proc.wait()
 return dict(city=e['city'],method=e['method'],exit_code=code)
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
 for future in concurrent.futures.as_completed([ex.submit(run,e) for e in entries if e['method']!='urban_dreamer']):
  r=future.result();results.append(r);print(json.dumps(r),flush=True)
  (OUT/'baseline_queue_status.json').write_text(json.dumps(dict(results=results,completed=len(results),total=33,elapsed_seconds=time.time()-start),indent=2)+'\n')
