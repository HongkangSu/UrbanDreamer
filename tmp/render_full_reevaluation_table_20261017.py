from pathlib import Path
import csv,json,argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path);args=parser.parse_args()
P=args.root or Path(Path('tmp/full_reevaluation_latest_path.txt').read_text().strip());d=json.loads((P/'verified_results.json').read_text());assert d['status']=='completed'
methods=['urban_dreamer','online_ac','gnac','ippo','mappo','iql','dreamer','mamba','mbpo','dta','shortest_ref','dynamic_shortest_ref']
labels=dict(zip(methods,['UrbanDreamer','Online AC','GNAC','IPPO','MAPPO','IQL','Dreamer','MAMBA','MBPO','DTA','Static shortest','Dynamic shortest']))
if any(r['method']=='ga_routing' for r in d['table']):
 methods.insert(methods.index('shortest_ref'),'ga_routing');labels['ga_routing']='GA (30 s)'
cities={'ingolstadt_custom':'Ingolstadt','coulommiers':'Coulommiers','montereau_fault_yonne':'Montereau-Fault-Yonne'}
keys=['all_demand_censored_att_seconds','completion_rate','all_demand_censored_p90_seconds','all_demand_censored_p95_seconds','branch_freeflow_excess_seconds_per_started_vehicle']
lookup={(r['city'],r['method']):r for r in d['table']};export=[];text=[]
fig,axes=plt.subplots(3,1,figsize=(15,18 if 'ga_routing' in methods else 17));fig.subplots_adjust(top=.94,bottom=.07,hspace=.22)
for ax,(city,label) in zip(axes,cities.items()):
 ax.axis('off');ax.set_title(label,fontsize=18,fontweight='bold',loc='left',pad=10)
 rows=[];text += [label,'Method | Checkpoint episode | ATT (s) | Completion (%) | P90 (s) | P95 (s) | Diversion cost (s/started vehicle)']
 for method in methods:
  r=lookup[city,method];v=[r[k]*(100 if k=='completion_rate' else 1) for k in keys];ep=r['checkpoint_episode'];cells=[labels[method],str(ep) if ep is not None else '--',*[f'{x:.2f}' for x in v]];rows.append(cells);text.append(' | '.join(cells))
  export.append(dict(city=label,method=labels[method],checkpoint_episode=ep,ATT_seconds=v[0],Completion_percent=v[1],P90_seconds=v[2],P95_seconds=v[3],Diversion_cost_seconds_per_started_vehicle=v[4]))
 table=ax.table(cellText=rows,colLabels=['Method','Checkpoint','ATT (s)','Completion (%)','P90 (s)','P95 (s)','Cost (s/veh)'],loc='center',cellLoc='center',colWidths=[.21,.11,.13,.15,.13,.13,.14]);table.auto_set_font_size(False);table.set_fontsize(10.5);table.scale(1,1.65)
 for (i,j),cell in table.get_celld().items():
  cell.set_edgecolor('#d5dce5')
  if i==0:cell.set_facecolor('#243c5a');cell.get_text().set_color('white');cell.get_text().set_weight('bold')
  elif i==1:cell.set_facecolor('#deecfa');cell.get_text().set_weight('bold')
  elif i%2==0:cell.set_facecolor('#f3f5f7')
 text.append('')
fig.suptitle('Fixed-seed evaluation: baselines and UrbanDreamer',fontsize=20,fontweight='bold')
fig.text(.5,.025,'Three paired demand seeds (10042–10044), demand seed 42; neural policies argmax; 3810 s horizon.\nATT/P90/P95 cover all demand from scheduled departure, including insertion wait and censored unfinished trips.\nCost: realized-turn free-flow opportunity cost per actually started vehicle. Different training budgets; PPO v7 excluded. GA:30s whole-route search; loops erased; fallback reported.',ha='center',fontsize=10)
for ext in ['png','pdf']:fig.savefig(P/f'comparison_table.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
with (P/'comparison_table.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(export[0]));w.writeheader();w.writerows(export)
(P/'comparison_table.txt').write_text('\n'.join(text)+'\n')
b=chr(92);lines=[b+'begin{tabular}{l'+'rrrrr'*3+'}',b+'hline','Method & '+' & '.join(b+'multicolumn{5}{c}{'+x+'}' for x in cities.values())+' '+b*2,' & '+' & '.join(['ATT & Completion & P90 & P95 & Cost']*3)+' '+b*2,b+'hline']
for method in methods:
 cells=[labels[method]]
 for city in cities:
  r=lookup[city,method];cells += [f"{r[k]*(100 if k=='completion_rate' else 1):.2f}" for k in keys]
 lines.append(' & '.join(cells)+' '+b*2)
lines += [b+'hline',b+'end{tabular}','% ATT/P90/P95: all-demand scheduled-departure censored seconds. Completion: percent.', '% Cost: seconds per started vehicle. Three paired demand seeds; checkpoint budgets differ (see CSV).']
(P/'comparison_table.tex').write_text('\n'.join(lines)+'\n')
print('\n'.join(text))
