exec(open('tmp/ppo_update_audit_20261016/probe.py').read().split('out=[]')[0])
from copy import deepcopy
import numpy as np
from types import SimpleNamespace
out=[]
for cls in [IppoMethod,MappoMethod]:
    model=method(cls)
    model.on_episode(*data(model),episode=1)
    rows,windows=data(model)
    for row in rows:
        row['features'][:]=row['features'][0]
        f,m=model._row_tensors([row])
        with torch.no_grad():
            row['old_log_prob']=float(torch.distributions.Categorical(logits=model.actor.forward_logits(f,m)).log_prob(torch.tensor([row['action']]))[0])
    targets,_,inputs=model._targets(rows,windows)
    model._targets=lambda *_args,result=(targets,torch.zeros(len(rows)),inputs):result
    before=deepcopy(model.actor.state_dict())
    probe_rows,_=data(model)
    probe_features,probe_mask=model._row_tensors(probe_rows)
    with torch.no_grad():
        probe_before=torch.distributions.Categorical(logits=model.actor.forward_logits(probe_features,probe_mask)).probs
    metrics=model.on_episode(rows,windows,episode=2)
    with torch.no_grad():
        probe_after=torch.distributions.Categorical(logits=model.actor.forward_logits(probe_features,probe_mask)).probs
    out.append(dict(algorithm=cls.__name__, max_probe_probability_change=float((probe_after-probe_before).abs().max()), max_parameter_change=max(float((v-before[k]).abs().max()) for k,v in model.actor.state_dict().items()), **{k:metrics[k] for k in ['actor_optimizer_steps','actor_gradient_norm','policy_kl']}))
print(json.dumps(out,indent=2))
Path('outputs/ppo_repair_20261016/audit_update/zero_gradient_adam.json').write_text(json.dumps(out,indent=2))
