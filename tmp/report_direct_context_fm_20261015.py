import json,hashlib
from pathlib import Path
root=Path.cwd();out=root/'outputs/direct_context_fm_20261014/v2_cpu4'
read=lambda p:json.loads(p.read_text())
a=read(out/'summary.json')
report=dict(status='complete_screening',experiment='Equal-budget directly conditioned small FM residuals with a completed frozen backbone',checkpoint_sha256='2f63f0b2a5fe18f3a3bd9be27ffd1e63a2755464e7f2944e1227334f0ffe6af6',completed_pretrain_wm_updates=49636,protocol=read(out/'protocol.json'),split_audit=read(out/'data_split_audit.json'),results=a,initial_gates=read(out/'initial_gates.json'),cache_compatibility=read(out/'parent_cache_compatibility.json'),interpretation=dict(soft_shift='Moving source AND reference to z+b remains worse than equal-budget zero source/ref mean z.',hint='Keeping source/ref mean z and exposing b as an extra input gives a small positive signal in ES/speed CRPS; count CRPS CI crosses zero.',necessity='Does not establish necessity of prior or benefit for routing policies.',optimization='Small-head validation patience met at95epochs; not evidence that fulljointmodel is optimized.',scope='Euler4 CPU screening only; no Euler100 equivalence. One headseed/frozen pretrainedbackbone, paired16 demandgroups/32states, K8.',h4='Neitherpriorarm passes prespecifiedjoint ES+bothoccupiedCRPS+massgate; no H4 experiment.'),artifacts=dict(root=str(out.relative_to(root)),summary=str((out/'summary.json').relative_to(root)),source=str((out/'source').relative_to(root))),validation=dict(tests='18 passed: direct-context and nonlinear residual tests',compile='passed',diff_check='passed'))
v=out/'independent_verification.json'
if v.exists():report['independent_verification']=read(v)
(root/'docs/direct_context_fm_20261014.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(output='docs/direct_context_fm_20261014.json',independent_verification=v.exists())))
