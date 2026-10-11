import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/home/wangwenrui/shk/UrbanDreamer/urban_dreamer')
SOURCE = ROOT/'outputs/ingolstadt_full_retrain_20261009/source_paper_reward_20261010'
BASE = ROOT/'outputs/ingo_cohort_reward_audit_20261011'
sys.path.insert(0, str(SOURCE))
import torch
from urban_dreamer.learning.episode_archive import read_episode_archive
from urban_dreamer.learning.global_reward import causal_context
from urban_dreamer.models.policy import CentralizedLatentValueCritic
from urban_dreamer.models.world_model import UrbanDreamerWorldModel
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.config import load_config
from urban_dreamer.workflow.pipeline import combined_input_hash
from tools.train_codec_ready import SparseGraphCodec

torch.set_num_threads(2)
torch.set_num_interop_threads(2)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

out=BASE/'critic_sensitivity.json'
if out.exists():raise FileExistsError(out)
cfg=load_config(BASE/'run_config.toml')
prepared=validate_prepared_inputs(cfg)
net=load_traffic_topology(prepared).network
checkpoint=torch.load(BASE/'checkpoint.pt',map_location='cpu',weights_only=False)
assert checkpoint['config_fingerprint']==cfg.semantic_fingerprint
assert checkpoint['input_hash']==combined_input_hash(prepared)
model=UrbanDreamerWorldModel(adjacency=torch.as_tensor(net.adjacency,dtype=torch.float32),
    static_features=torch.as_tensor(net.static_features,dtype=torch.float32),
    low_frequency_rank=cfg.method.low_frequency_rank,prior_noise_scale=cfg.method.prior_noise_scale,
    event_protocol=True,codec_speed_positive=cfg.codec.speed_positive)
model.load_state_dict(checkpoint['training_state']['models']['world_model'],strict=True)
model.flow_projector.__class__=SparseGraphCodec
model.flow_projector.register_buffer('sparse_adjacency',torch.as_tensor(net.adjacency).to_sparse().coalesce(),persistent=False)
model.eval().requires_grad_(False)
trained=CentralizedLatentValueCritic()
trained.load_state_dict(checkpoint['training_state']['models']['latent_critic'],strict=True)
trained.eval().requires_grad_(False)
del checkpoint
initial_path=ROOT/'outputs/ingolstadt_full_retrain_20261009/online_seed/checkpoints/latest.pt'
initial_payload=torch.load(initial_path,map_location='cpu',weights_only=False)
initial=CentralizedLatentValueCritic()
initial.load_state_dict(initial_payload['training_state']['models']['latent_critic'],strict=True)
initial.eval().requires_grad_(False)
initial_next=initial_payload['next_unit']
del initial_payload
path=BASE/'frozen_policy_real/episodes/evaluation/episode_000000.npz'
archive=read_episode_archive(path,expected_session_protocol='event')
assert archive['metadata']['input_hashes']==prepared.input_hashes
def measure(critic,z,context):
    c=context.detach().clone().requires_grad_(True)
    value=critic(z,c).reshape(())
    derivative=torch.autograd.grad(value,c)[0][0]
    with torch.no_grad():
        x=torch.cat((z.mean(-2),c),-1)
        h1=critic.trunk[1](critic.trunk[0](x))
        h2=critic.trunk[3](critic.trunk[2](h1))
    return dict(value=value.item(),context_gradient=derivative.tolist(),
        dV_d_elapsed_fraction=derivative[0].item(),
        layer1_tanh_saturation_fraction=(h1.abs()>.99).float().mean().item(),
        layer2_tanh_saturation_fraction=(h2.abs()>.99).float().mean().item())

result=dict(checkpoint_sha256=sha(BASE/'checkpoint.pt'),initial_checkpoint_sha256=sha(initial_path),
    initial_checkpoint_next_unit=initial_next,archive_sha256=sha(path),source_dir=str(SOURCE),
    protocol='Fixed real encoded latent, counterfactual context sensitivity; model parameters frozen.',
    limitations=['Context sweeps are not physically realized states; local derivatives/saturation do not prove training failure cause.',
        'Initial critic sees the trained codec latent used for trained critic; this isolates critic weights, not historical initial end-to-end behavior.',
        'Saturation fractions are descriptive at these inputs, not a causal attribution or optimization experiment.'],rows=[])
for index in (10,30,60,80):
    state=archive['states'][index]
    active=sum(v['departed'] and not v['arrived'] for v in state['vehicles'])
    count=state['started_vehicle_count']
    context=torch.as_tensor(causal_context(state['time'],archive['deadline_seconds'],active,count,state['cumulative_vehicle_seconds']))[None]
    with torch.no_grad():z=model.encode(torch.as_tensor(archive['flow'][index:index+1]))
    pooled=z.mean(-2)
    row=dict(seed_index=index,time=state['time'],raw_context=context[0].tolist(),
        latent_pooled_rms=pooled.square().mean().sqrt().item(),latent_pooled_max_abs=pooled.abs().max().item(),
        latent_all_rms=z.square().mean().sqrt().item(),latent_all_max_abs=z.abs().max().item(),critics={})
    for name,critic in [('trained',trained),('initial',initial)]:
        values=measure(critic,z,context)
        sweep=[]
        for elapsed in (0.,.25,.5,.75,1.):
            altered=context.clone();altered[0,0]=elapsed
            sweep.append(dict(elapsed_fraction=elapsed,**measure(critic,z,altered)))
        altered=context.clone();altered[0,2]=0.
        zero=measure(critic,z,altered)
        values.update(elapsed_fraction_sweep=sweep,zero_cumulative_time=zero,
            zero_cumulative_value_change=zero['value']-values['value'])
        row['critics'][name]=values
    result['rows'].append(row)
    print(json.dumps(row),flush=True)
out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
