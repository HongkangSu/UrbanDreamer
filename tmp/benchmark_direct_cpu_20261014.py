import json
from pathlib import Path
import sys
import time

root = Path.cwd()
sys.path.insert(0, str(root / 'outputs/direct_context_fm_20261014/v1/source'))
import torch
from tools.component_experiment_data import load_frozen
from tools.train_direct_context_fm import DirectResidual, MeanOverridePrior
from urban_dreamer.learning.pretrain_cache import CompactPretrainDataset
from tools.world_model_evaluation import _repeat_events

torch.set_num_threads(4)
torch.set_num_interop_threads(2)
torch.manual_seed(41)
heads = [DirectResidual() for _ in range(3)]
optimizers = [torch.optim.AdamW(h.parameters(), lr=.001) for h in heads]
x, y = torch.randn(32768, 449), torch.randn(32768, 64)
elapsed = []
for iteration in range(4):
    started = time.perf_counter()
    for h, optimizer in zip(heads, optimizers):
        optimizer.zero_grad(set_to_none=True)
        loss = (h(x) - y).square().mean()
        loss.backward()
        optimizer.step()
    elapsed.append(time.perf_counter() - started)
print(json.dumps(dict(stage='head_benchmark', three_arm_step_seconds=elapsed)), flush=True)
config, prepared, topology, model, _ = load_frozen(root / 'outputs/component_experiments_20261011', device='cpu')
model.eval().requires_grad_(False)
model.set_online_chunk_profile()
model.dynamics.graph_prior = MeanOverridePrior(model.dynamics.graph_prior, zero=True)
split = json.loads((root / 'outputs/direct_context_fm_20261014/v1/split.json').read_text())
data = CompactPretrainDataset(config, prepared, topology, 5, cache_manifest_path=Path(split['cache_manifest_path']), split='validation')
item = split['validation_conditions'][0]
batch = data.batch_from_starts(item['episode_index'], [item['start']], fm_only=False)
with torch.inference_mode():
    z = model.encode(batch.flow[:, 0]).repeat(4, 1, 1)[:, None]
    actions = batch.action[:, :1].repeat(4, 1, 1)
    events = _repeat_events(batch.action_events.for_step(0), 4)
    condition = model.dynamics.encode_condition(actions, action_events=events)
    samples = []
    for iteration in range(4):
        started = time.perf_counter()
        out = model.dynamics(actions, packed_latents=z, flow_time=torch.full((4, 1), .5), condition=condition)
        float(out.latent_velocity.sum())
        samples.append(time.perf_counter() - started)
result = dict(threads=4, head_steps_seconds=elapsed, batch4_dynamics_seconds=samples,
              measured_on_validation_only=True, model_parameters_updated=False)
(root / 'outputs/direct_context_fm_20261014/v1/cpu_timing_probe.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
