import json, pathlib, statistics, math
base=pathlib.Path('outputs/city_baselines_static_round2_2000_20261015_173045')
out=pathlib.Path('outputs/ppo_repair_20261016/audit_city');out.mkdir(parents=True,exist_ok=True)
keys='completion_rate global_reward_sum branch_entropy raw_advantage_mean raw_advantage_std raw_advantage_positive_fraction policy_kl policy_kl_max clip_fraction actor_optimizer_steps critic_optimizer_steps critic_mse_before critic_mse_after critic_explained_variance_before critic_explained_variance_after critic_bias_before critic_bias_after actor_gradient_norm critic_gradient_norm'.split()
def stats(rows):
 return {k:dict(mean=statistics.mean(r[k] for r in rows),min=min(r[k] for r in rows),max=max(r[k] for r in rows)) for k in keys if all(k in r for r in rows)}|{'n':len(rows),'critic_epoch_zero_count':sum(r.get('critic_best_epoch')==0 for r in rows),'kl_stop_count':sum(r.get('kl_early_stopped',False) for r in rows)}
report={}
for city in ['ingolstadt_custom','coulommiers','montereau_fault_yonne']:
 for alg in ['ippo','mappo']:
  records=[json.loads(l) for l in open(base/city/alg/'metrics.jsonl')];train=[r for r in records if r['phase']=='train'];by_ep={r['episode']:r for r in train}
  item={'all':stats(train),'last20':stats(train[-20:]),'evaluation':{str(ep):{k:statistics.mean(r[k] for r in records if r['phase']=='evaluation' and r['episode']==ep) for k in ['completion_rate','global_reward_sum','branch_entropy']} for ep in sorted({r['episode'] for r in records if r['phase']=='evaluation'})},'attempts':[]}
  for p in sorted((base/'attempts'/f'{city}_{alg}').glob('*/metrics_before.jsonl')):
   rr=[json.loads(l) for l in open(p)];tr=[r for r in rr if r['phase']=='train'];diff=[r['episode'] for r in tr if r['episode'] in by_ep and r!=by_ep[r['episode']]]
   item['attempts'].append({'path':str(p),'train_rows':len(tr),'last_episode':max((r['episode'] for r in tr),default=0),'different_episodes':diff})
  if city=='ingolstadt_custom' and alg=='ippo':
   item['windows']={f'{a}-{b}':stats([r for r in train if a<=r['episode']<=b]) for a,b in [(160,179),(180,184),(185,220),(620,639),(640,649),(650,662)]}
   item['transition_rows']=[{k:r.get(k) for k in ['episode']+keys+['critic_best_epoch','kl_early_stopped']} for r in train if 175<=r['episode']<=220 or 635<=r['episode']<=662]
  item['zero_critic_intervals']=[]
  for r in train:
   if r['critic_best_epoch']==0:
    if item['zero_critic_intervals'] and item['zero_critic_intervals'][-1][1]==r['episode']-1:item['zero_critic_intervals'][-1][1]=r['episode']
    else:item['zero_critic_intervals'].append([r['episode'],r['episode']])
  report[f'{city}/{alg}']=item
(out/'metrics_evidence.json').write_text(json.dumps(report,indent=2))
for name,item in report.items():
 print(name, 'n',item['all']['n'],'rollback',item['all']['critic_epoch_zero_count'],'last20_EV',item['last20']['critic_explained_variance_after']['mean'],'KLmax',item['all']['policy_kl_max']['max'])
 print('zero intervals',item['zero_critic_intervals'][-10:]);print('attempts',[(x['path'].split('/')[-2],x['last_episode'],len(x['different_episodes'])) for x in item['attempts']])
 print('eval',item['evaluation'])
