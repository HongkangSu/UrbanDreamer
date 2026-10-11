"""Private, bounded codec probe; never installs weights into a running model."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.train_codec_ready import SparseGraphCodec
from urban_dreamer.learning.codec_training import codec_objective
from urban_dreamer.learning.cohort_targets import prepare_cohort_batch
from urban_dreamer.learning.pretrain_cache import CompactPretrainDataset
from urban_dreamer.learning.offline_data import _identity
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.config import load_config


def masks(flow, following, capacities, limits):
    count = flow[..., 2] * capacities
    mean_speed = .5 * (flow[..., 1] + following[..., 1]) * limits
    return count > 0, (count >= 5 - 1e-5) & (mean_speed <= 2)


def masked_mean(value, mask):
    return value[mask].mean() if bool(mask.any()) else value.sum() * 0


def weighted_speed_objective(decoded, flow, following, capacities, limits):
    """Diagnostic extra terms only; production codec loss is untouched."""
    occupied, slow = masks(flow, following, capacities, limits)
    error = F.smooth_l1_loss(decoded[..., 1], .5 * (flow[..., 1] + following[..., 1]),
                             beta=.1, reduction='none')
    return masked_mean(error, occupied) + masked_mean(error, slow)


def scores(codec, flow, following, capacities, limits):
    with torch.no_grad():
        decoded = codec.decode(codec(flow))
        occupied, slow = masks(flow, following, capacities, limits)
        error = (decoded[..., 1] - .5 * (flow[..., 1] + following[..., 1])) * limits
        identity_error = (decoded[..., 1] - flow[..., 1]) * limits
        count_error = (decoded[..., 2]-flow[..., 2])*capacities
        target_total = (flow[...,2]*capacities).sum(-1)
        total_error = count_error.sum(-1)
        def mean(value, mask):
            return float(value[mask].mean()) if bool(mask.any()) else None
        return dict(slow_roads=int(slow.sum()), occupied_roads=int(occupied.sum()),
                    slow_speed_mae_mps=mean(error.abs(), slow),
                    slow_same_frame_speed_mae_mps=mean(identity_error.abs(), slow),
                    occupied_speed_mae_mps=mean(error.abs(), occupied),
                    volume_mae_vehicles=float(count_error.abs().mean()),
                    occupied_volume_mae_vehicles=mean(count_error.abs(),occupied),
                    total_volume_mae_vehicles=float(total_error.abs().mean()),
                    total_volume_bias_vehicles=float(total_error.mean()),
                    total_volume_relative_error=float((total_error.abs()/target_total.clamp_min(1.)).mean()))


def freeze_encoder_for_decoder_probe(codec):
    """Explicitly permit the four decoder modules and nothing else."""
    allowed=('decoder_shared.','decoder_speed.','decoder_volume.','decoder_other.')
    for name, parameter in codec.named_parameters():
        parameter.requires_grad_(name.startswith(allowed))


def frozen_state_hash(codec):
    digest=hashlib.sha256()
    for name,parameter in codec.named_parameters():
        if not parameter.requires_grad:
            value=parameter.detach().cpu().contiguous()
            digest.update(name.encode()+str(value.dtype).encode()+str(tuple(value.shape)).encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path, result):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parent = ROOT/'outputs/ingolstadt_noincident_pretrain_20261008'
    parser.add_argument('--accuracy-dir', type=Path, default=parent/'world_accuracy_20261009')
    parser.add_argument('--output-dir', type=Path, default=parent/'slow_queue_diagnosis_20261009')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--updates', type=int, default=200)
    args = parser.parse_args()
    if not 0 <= args.updates <= 400:
        parser.error('private probe updates must be within [0, 400]')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(.08, device)
    config = load_config(args.accuracy_dir/'online_snapshot/run_config.toml')
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    network = topology.network
    capacities = torch.as_tensor(network.capacities, device=device).float().clamp_min(1.)
    limits = torch.as_tensor(network.speed_limits_mps, device=device).float()
    checkpoint_path = args.accuracy_dir/'online_snapshot/checkpoints/latest.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    codec = SparseGraphCodec(adjacency=torch.tensor(network.adjacency), positive_speed=config.codec.speed_positive)
    codec.register_buffer('sparse_adjacency', torch.tensor(network.adjacency).to_sparse().coalesce(), persistent=False)
    codec.load_state_dict({key.removeprefix('flow_projector.'): value
                           for key, value in checkpoint['training_state']['models']['world_model'].items()
                           if key.startswith('flow_projector.')})
    codec.to(device).eval()
    accuracy = json.loads((args.accuracy_dir/'results.json').read_text())
    cases = [c for c in accuracy['recursive_cases'] if c['method'] in ('online_ac','random','congested')]
    manifest_path = Path(config.data.validation_cache_manifest or config.data.pretrain_cache_manifest)
    manifest = json.loads(manifest_path.read_text())
    paths = {c['path'] for c in cases}
    selected = {**manifest, 'entries':[e for e in manifest['entries'] if e['path'] in paths],
                'sources':[e for e in manifest['sources'] if e['path'] in paths]}
    selected.pop('identity')
    selected['identity'] = _identity(selected)
    mini_path = args.output_dir/'probe_cache_manifest.json'
    atomic_json(mini_path, selected)
    dataset = CompactPretrainDataset(config, prepared, topology, 9, cache_manifest_path=mini_path,
                                     split='validation')
    initial, train_inputs, train_next, heldout_inputs, heldout_next = [], [], [], [], []
    evaluation_frames=[]
    started = time.monotonic()
    for case in cases:
        index = next(i for i,e in enumerate(dataset.manifest['entries']) if e['path']==case['path'])
        batch = dataset.batch_from_starts(index,[case['actual_start']],fm_only=False)
        scoped = prepare_cohort_batch(batch, capacities=capacities.cpu())
        raw, hybrid = batch.flow.to(device), scoped.transition_flow.to(device)
        for step in (1,3,8):
            for kind, flow in [('raw',raw),('hybrid_C0',hybrid)]:
                current, following = flow[:,step], flow[:,step+1]
                occupied, slow = masks(current, following, capacities, limits)
                with torch.no_grad(): decoded = codec.decode(codec(current))
                rows=[]
                for edge in slow[0].nonzero().flatten().tolist():
                    rows.append(dict(edge_index=edge, edge_id=network.edge_ids[edge],
                        current_speed_mps=float(current[0,edge,1]*limits[edge]),
                        following_speed_mps=float(following[0,edge,1]*limits[edge]),
                        target_mean_speed_mps=float(.5*(current[0,edge,1]+following[0,edge,1])*limits[edge]),
                        predicted_speed_mps=float(decoded[0,edge,1]*limits[edge]),
                        full_count=float(raw[0,step,edge,2]*capacities[edge]),
                        C0_count=float(hybrid[0,step,edge,2]*capacities[edge])))
                initial.append(dict(case=case['method'], horizon_seconds=step*30,input_kind=kind,
                                    **scores(codec,current,following,capacities,limits), roads=rows))
                evaluation_frames.append((case['method'],step,kind,current,following))
                if step in (1,8):
                    train_inputs.append(current); train_next.append(following)
        for step in (2,4,7):
            for flow in (raw,hybrid):
                heldout_inputs.append(flow[:,step]); heldout_next.append(flow[:,step+1])
    train, following = torch.cat(train_inputs), torch.cat(train_next)
    heldout, heldout_following = torch.cat(heldout_inputs), torch.cat(heldout_next)
    decoded = codec.decode(codec(train))
    speed_loss = F.smooth_l1_loss(decoded[...,1],.5*(train[...,1]+following[...,1]),beta=.1)
    speed_grad = torch.autograd.grad(speed_loss, decoded, retain_graph=True)[0][...,1]
    occupied, slow = masks(train,following,capacities,limits)
    loss, _ = codec_objective(codec,train,following,network,config.method)
    loss.backward()
    group_norms={}
    for group,prefixes in dict(encoder=('input_proj','gnn_layers'), decoder_speed=('decoder_speed',),
                               decoder_shared=('decoder_shared',)).items():
        grads=[p.grad for name,p in codec.named_parameters() if name.startswith(prefixes) and p.grad is not None]
        group_norms[group]=dict(l2=float(torch.stack([g.square().sum() for g in grads]).sum().sqrt()),
                               finite=all(bool(torch.isfinite(g).all()) for g in grads))
    gradient=dict(speed_loss=float(speed_loss), slow_cells=int(slow.sum()),empty_cells=int((~occupied).sum()),
        slow_abs_grad_mean=float(masked_mean(speed_grad.abs(),slow)),
        empty_abs_grad_mean=float(masked_mean(speed_grad.abs(),~occupied)),
        slow_abs_grad_sum=float(speed_grad[slow].abs().sum()),
        empty_abs_grad_sum=float(speed_grad[~occupied].abs().sum()),parameter_groups=group_norms)
    result=dict(status='extraction_complete',checkpoint=str(checkpoint_path),
                checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                source_cache_manifest_identity=manifest['identity'],production_modified=False,
        train_frames=len(train), heldout_nearby_frames=len(heldout), initial=initial,
        nearby_heldout_before=scores(codec,heldout,heldout_following,capacities,limits),
                gradient=gradient, fits=[], notes=['Private cloned weights only.',
                'Nearby heldout frames share episodes and are not independent population validation.',
                'Same-frame error excludes future speed-mean label ambiguity.'])
    result['diagnostic_weighting']=dict(original_speed_weight=getattr(config.method,'codec_speed_weight',1.),
        extra_occupied_speed_weight=1., extra_slow_speed_weight=1., smooth_l1_beta=.1,
        optimization_speed_units='speed/speed_limit ratio; metrics converted to m/s',
        speed_label='mean of current and following endpoint normalized speed')
    atomic_json(args.output_dir/'probe_results.json',result)
    print(json.dumps(dict(stage='extraction_complete',gradient=gradient,initial=initial)),flush=True)
    for mode in ('original','occupied_and_slow_weighted','frozen_encoder_decoder_weighted'):
        candidate=copy.deepcopy(codec).train()
        frozen_hash=None
        if mode=='frozen_encoder_decoder_weighted':
            freeze_encoder_for_decoder_probe(candidate)
            frozen_hash=frozen_state_hash(candidate)
        optimizer=torch.optim.AdamW([p for p in candidate.parameters() if p.requires_grad],
                                   lr=config.codec.learning_rate,weight_decay=.01)
        before=scores(candidate,train,following,capacities,limits)
        history=[]
        for update in range(args.updates):
            optimizer.zero_grad(set_to_none=True)
            objective,_=codec_objective(candidate,train,following,network,config.method)
            if mode!='original':
                objective=objective+weighted_speed_objective(candidate.decode(candidate(train)),train,following,capacities,limits)
            objective.backward()
            norm=torch.nn.utils.clip_grad_norm_(candidate.parameters(),10.,error_if_nonfinite=True)
            optimizer.step()
            if (update+1)%50==0: history.append(dict(update=update+1,loss=float(objective),gradient_norm=float(norm)))
        result['fits'].append(dict(mode=mode, updates=args.updates, before=before,
            train_after=scores(candidate,train,following,capacities,limits),
            nearby_heldout_after=scores(candidate,heldout,heldout_following,capacities,limits),history=history))
        if frozen_hash is not None:
            after_hash=frozen_state_hash(candidate)
            if after_hash!=frozen_hash:
                raise AssertionError('frozen encoder or t_all parameters changed')
            result['fits'][-1].update(frozen_parameters_sha256_before=frozen_hash,
                                     frozen_parameters_sha256_after=after_hash,
                                     frozen_parameters_exactly_unchanged=True)
        result['fits'][-1]['by_case_after']=[dict(case=name,horizon_seconds=step*30,input_kind=kind,
            **scores(candidate,frame,next_frame,capacities,limits))
            for name,step,kind,frame,next_frame in evaluation_frames]
        atomic_json(args.output_dir/'probe_results.json',result)
    result.update(status='complete',elapsed_seconds=time.monotonic()-started)
    atomic_json(args.output_dir/'probe_results.json',result)
    print(json.dumps(result['fits']),flush=True)


if __name__=='__main__': main()
