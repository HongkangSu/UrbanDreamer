import sys, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / 'tests'))
import torch
from test_routing_ppo_baselines import data, method
from urban_dreamer.learning.baselines.ppo import IppoMethod, MappoMethod
torch.set_num_threads(1)
out=[]
for cls in [IppoMethod, MappoMethod]:
    for lr, budget, batch in [(3e-4, 1e-8, 4), (.1, .01, 4), (3e-4, .01, 1)]:
        model=method(cls)
        model.target_kl=budget
        model.minibatch_size=batch
        for g in model.actor_optimizer.param_groups: g['lr']=lr
        metrics=model.on_episode(*data(model), episode=1)
        out.append(dict(algorithm=cls.__name__, lr=lr, target_kl=budget, batch=batch,
                        **{k:metrics[k] for k in ['policy_kl','policy_kl_max','actor_optimizer_steps','kl_early_stopped']}))
print(json.dumps(out,indent=2))
Path('outputs/ppo_repair_20261016/audit_update/actual_adam_kl.json').write_text(json.dumps(out,indent=2))
