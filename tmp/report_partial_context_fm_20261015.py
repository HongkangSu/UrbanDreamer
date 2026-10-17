from pathlib import Path
import json
root=Path.cwd();out=root/'outputs/partial_context_fm_20261015/v2_gpu0'
read=lambda p:json.loads(p.read_text())
assert (out/'validation_selection.json').exists()
a=read(out/'summary.json');v=read(out/'independent_verification.json')
assert a['status']=='complete' and v['status']=='pass'
p=read(out/'protocol.json');sel=a['selection'];metrics=['energy_score','energy_score_u','count_crps_occupied','speed_crps_occupied','count_crps_u_occupied','speed_crps_u_occupied','count_mae_occupied','speed_mae_occupied','count_coverage90_occupied','speed_coverage90_occupied','count_width90_occupied','speed_width90_occupied','total_count_absolute_error']
result=dict(status='complete',experiment='Equal-budget partial FM adaptation separating coarse source mean and reference state',checkpoint_sha256=p['reference_hashes']['checkpoint'],completed_pretrain_wm_updates=49636,
 prior='Fixed state-only weak softtau.1 predictor fitted on HYBRIDCt labels; originalrank16source Gaussian noise paired',protocol=p,
 interpretation=dict(source_choice='C1 remains substantially worse than B after equal partial adaptation.',joint_choice='C2 reduces the mismatch relative to C1 but remains worse than B in ES/countCRPS and totalcount error; speedCRPS difference CI includeszero.',budget='C2 still improves at last validation; cap reached, cannot claim fullyoptimized or universally useless coarse sources.',coverage='Empirical K8 nominal90 intervals under-cover on these states; finite ensemble quantiles also affect coverage. No calibrated-generative-superiority claim.'),selection=sel,results={r['arm']:{m:r['metrics'][m] for m in metrics if m in r['metrics']} for r in a['test']['arms']},
 contrasts={k:{m:value[m] for m in metrics if m in value} for k,value in a['pairs'].items()},
 validation_curve=[dict(epoch=0,ES={r['arm']:r['metrics']['energy_score']['estimate'] for r in read(out/'validation/epoch000_summary.json')['arms']})]+[dict(epoch=r['epoch'],ES={k:v['ES'] for k,v in r['integrated_validation'].items()},train_CFM_MSE=r['train_CFM_MSE'],epoch_seconds=r['epoch_seconds']) for r in read(out/'training_history.json')],
 scope=['All three arms receive the same background hint; this tests incremental value of coarse source/reference choices after equal adaptation.',
 'Separately adapted models; not a within-identical-weights source-only intervention.',
 'Original pretrained backbone complete; partial adaptation is budget-limited if cap reached before patience.',
 'One frozen pretrained checkpoint and adaptation seed, one city; no policy or cross-city benefit claim.',
 'No SUMO interaction or complete joint-model retraining was performed.'],
 resource_restart=read(root/'outputs/partial_context_fm_20261015/v1/stop_reason.json'),
 runtime=dict(elapsed_seconds=a['elapsed_seconds'],max_gpu_allocated_mib=a['max_gpu_allocated_mib']),
 independent_verification=v,validation=dict(focused_tests='28 passed',compile='passed',diff_check='passed'),
 artifacts=dict(root=str(out.relative_to(root)),summary=str((out/'summary.json').relative_to(root)),pilot='outputs/partial_context_fm_20261015/pilot_v1'))
if (out/'h4_gate.json').exists():result['h4_gate']=read(out/'h4_gate.json')
if (out/'h4_followup/summary.json').exists():result['h4_followup']=read(out/'h4_followup/summary.json')
(root/'docs/partial_context_fm_20261015.json').write_text(json.dumps(result,indent=2)+'\n')
print('docs/partial_context_fm_20261015.json')
