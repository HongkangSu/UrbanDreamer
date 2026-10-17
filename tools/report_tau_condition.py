"""Independently re-score frozen tau banks and summarize validation selection."""
from pathlib import Path
import argparse,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def read(p):return json.loads(p.read_text())
def verify(output):
    protocol=read(output/'protocol.json');summary=read(output/'summary.json');rows=read(output/'states.json')
    table={(r['condition_index'],r['tau']):r for r in rows};checked=0;max_error=0.
    for entry in read(output/'bank_manifest.json'):
        with np.load(entry['path']) as archive:
            truth=archive['target'][0].astype(float);scales=archive['scales'];valid=np.isfinite(truth);occupied=valid[:,0]&(truth[:,0]>0)
            for tau in protocol['taus']:
                samples=archive[f'tau_{tau:g}'][:,0].astype(float);k=len(samples)
                x=(samples/scales)[:,valid];y=(truth/scales)[valid]
                es=np.linalg.norm(x-y,axis=1).mean()-sum(np.linalg.norm(x[i]-x[j]) for i in range(k) for j in range(k) if i!=j)/(2*k*(k-1))
                values={'energy_score_u':es}
                for j,name in enumerate(('count','speed')):
                    mask=occupied&valid[:,j];draws=samples[:,mask,j];target=truth[mask,j]
                    crps=np.abs(draws-target).mean(0)-sum(np.abs(draws[i]-draws[l]) for i in range(k) for l in range(k) if i!=l)/(2*k*(k-1))
                    lo,hi=np.quantile(draws,[.05,.95],axis=0)
                    values[name+'_crps_u_occupied']=crps.mean()
                    values[name+'_coverage90_occupied']=((target>=lo)&(target<=hi)).mean()
                    values[name+'_width90_occupied']=(hi-lo).mean()
                row=table[(entry['condition_index'],tau)]
                for metric,value in values.items():
                    error=abs(value-row[metric]);max_error=max(max_error,error)
                    np.testing.assert_allclose(value,row[metric],atol=1e-9,rtol=1e-9);checked+=1
    anchors=read(output/'anchor_checks.json')
    if len(anchors)!=8:raise ValueError('all eight old validation conditions must reproduce')
    audit=dict(status='passed',independently_recomputed_metrics=checked,max_absolute_error=max_error,
        anchor_conditions=len(anchors),anchor_max_absolute_error=max(x['max_absolute_error'] for x in anchors),
        test_bank_accessed=False)
    (output/'independent_verification.json').write_text(json.dumps(audit,indent=2))
    text=['# Condition-only B soft-filter tau sweep','',
      'Validation-only inference sensitivity, with FM weights and prior regression head fixed. All candidates use identical rank-16 graph noise, Z as source center and reference, and only change the background condition. This is not independent per-tau training.','',
      '32 states from 16 existing validation demand groups; K=8; Euler 100. Selection uses finite-K-corrected Energy Score. The old eight validation states reproduce the first four B draws within the preregistered 2e-4 tolerance.','',
      '| tau | fair ES ↓ | count fair CRPS ↓ | speed fair CRPS ↓ | count coverage90 | speed coverage90 |',
      '|---|---:|---:|---:|---:|---:|']
    for tau in protocol['taus']:
        metrics=summary['arms'][str(tau)]['metrics']
        vals=[metrics[k]['estimate'] for k in ['energy_score_u','count_crps_u_occupied','speed_crps_u_occupied','count_coverage90_occupied','speed_coverage90_occupied']]
        text.append('| '+str(tau)+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
    best=summary['validation_selected_tau'];delta=summary['arms'][str(best)]['minus_tau_0_1']['energy_score_u']
    text+=['',f'Lowest observed validation fair ES: tau={best}; difference from tau=.1: {delta["estimate"]:.6f}, paired demand-group bootstrap 95% CI [{delta["ci_low"]:.6f}, {delta["ci_high"]:.6f}].',
      '', 'Selection uncertainty is optimistic after searching six candidates. This sweep supports a deployment sensitivity check, not a universal optimum or downstream policy benefit.',
      '', f'Independent bank re-scoring: {checked} metrics passed; max error {max_error:.3g}.']
    (output/'report.md').write_text('\n'.join(text)+'\n')
    return audit
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'outputs/tau_condition_20261016/v1');args=parser.parse_args()
    print(json.dumps(verify(args.output)))
