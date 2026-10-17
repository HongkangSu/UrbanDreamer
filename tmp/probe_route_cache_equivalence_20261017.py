import sys,os,json,sqlite3,importlib.util,time
from pathlib import Path
root=Path.cwd();base=root/'outputs/all_baselines_fixed_metrics_20261017_060627';sys.path.insert(0,str(base/'sources/0'));os.environ.pop('URB_ROUTE_CACHE_DIR',None)
from urban_dreamer.traffic.route_candidates import RouteCandidates as New
from urban_dreamer.traffic.shared_route_cache import network_identity
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.config import load_config
spec=importlib.util.spec_from_file_location('urban_dreamer.traffic.route_candidates_original_probe',base/'sources/1/urban_dreamer/traffic/route_candidates.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);Old=mod.RouteCandidates
result=[]
for city in ['ingolstadt_custom','coulommiers','montereau_fault_yonne']:
 config=load_config(base/'sources/1/configs'/f'{city}.toml');network=load_traffic_topology(validate_prepared_inputs(config)).network;identity=network_identity(network);path=root/'outputs/urbandreamer_current_to_dest_1000_fp16_w8_r8_p2_20261013'/city/'shared_route_cache_200000'/f'{identity}_c200000.sqlite3';db=sqlite3.connect(f'file:{path}?mode=ro',uri=True);n=db.execute('select count(*) from routes').fetchone()[0];rows=[]
 for frac in [0,.25,.5,.75,1.]:rows+=db.execute('select current,destination,paths from routes order by sequence limit 8 offset ?', (max(0,min(n-8,int(frac*n))),)).fetchall()
 unique={(a,b):p for a,b,p in rows};old=Old(network,traffic_model=config.simulation.traffic_model);new=New(network,traffic_model=config.simulation.traffic_model);start=time.monotonic(); mismatches=[]
 for (a,b),payload in unique.items():
  expected=tuple(tuple(x) for x in json.loads(payload));v1=old.generate_paths(a,b);v2=new.generate_paths(a,b)
  if v1!=v2 or v1!=expected:mismatches.append([a,b])
 row=dict(city=city,identity=identity,cache_path=str(path),cached_entries=n,samples=len(unique),mismatches=mismatches,seconds=time.monotonic()-start);result.append(row);print(json.dumps(row),flush=True);db.close()
(root/'tmp/route_cache_equivalence_20261017.json').write_text(json.dumps(result,indent=2)+'\n');assert all(not r['mismatches'] for r in result)
