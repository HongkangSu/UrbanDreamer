"""Isolated callback benchmark; never mutates a training actor or checkpoint."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from urban_dreamer.models.policy import NextHopActor, set_shortest_path_prior


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--callbacks', type=int, default=1912)
    parser.add_argument('--rows', type=int, default=15)
    parser.add_argument('--candidates', type=int, default=3)
    parser.add_argument('--prior-weight', type=float, default=0.)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--gpu-ready-file', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(7319)
    actor = NextHopActor().eval().requires_grad_(False)
    if args.checkpoint is not None:
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        actor.load_state_dict(payload['training_state']['models']['actor'])
        del payload
    set_shortest_path_prior(actor, args.prior_weight, 30.)
    features = torch.rand(args.rows, args.candidates, 10)
    features[..., 4] *= .5
    mask = torch.ones(args.rows, args.candidates, dtype=torch.bool)
    if args.candidates > 2:
        mask[::3, -1] = False
    device = torch.device(args.device)
    result = dict(callbacks=args.callbacks, rows=args.rows, candidates=args.candidates,
                  device=str(device), synthetic_conditions=True, optimizer_steps=0,
                  timings={}, seed=7319, prior_weight=args.prior_weight, checkpoint=str(args.checkpoint) if args.checkpoint else None)
    gpu_actor = None
    if device.type == 'cuda':
        if args.gpu_ready_file is not None:
            print('CPU checkpoint load complete; waiting for GPU handoff', flush=True)
            while not args.gpu_ready_file.exists():
                time.sleep(.5)
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(.02, device)
        gpu_actor = copy.deepcopy(actor).to(device)

    layers = (actor.candidate_scorer[0], actor.candidate_scorer[2], actor.policy_head)
    weights = [(layer.weight.detach().numpy().T, layer.bias.detach().numpy()) for layer in layers]
    np_features = features.numpy()
    np_mask = mask.numpy()

    def numpy_logits():
        hidden = np_features
        for weight, bias in weights[:-1]:
            hidden = np.tanh(hidden @ weight + bias)
        logits = (hidden @ weights[-1][0] + weights[-1][1]).squeeze(-1)
        return torch.from_numpy(np.where(np_mask, logits, -np.inf))

    def sync() -> None:
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def measure(name, fn):
        for _ in range(20):
            fn()
        sync()
        start = time.perf_counter()
        for _ in range(args.callbacks):
            fn()
        sync()
        elapsed = time.perf_counter() - start
        result['timings'][name] = {'seconds': elapsed, 'us_per_callback': elapsed / args.callbacks * 1e6}

    with torch.no_grad():
        for threads in (1, 4):
            torch.set_num_threads(threads)
            measure(f'cpu_logits_threads{threads}', lambda: actor.forward_logits(features, mask))
            if gpu_actor is not None:
                def hybrid():
                    logits = actor.forward_logits(features, mask).to(device)
                    distribution = Categorical(logits=logits)
                    selected = distribution.sample()
                    selected.cpu().tolist()
                    return selected
                measure(f'cpu_logits_gpu_categorical_threads{threads}', hybrid)
        torch.set_num_threads(1)
        if args.prior_weight == 0:
            measure('numpy_logits', numpy_logits)
            result['numpy_legal_logit_max_abs_error'] = float((numpy_logits()[mask] - actor.forward_logits(features, mask)[mask]).abs().max())
        if gpu_actor is not None:
            def existing():
                selected = gpu_actor.sample_action(features.to(device), mask.to(device))
                selected.indices_and_entropy()
                return selected.candidate_index
            measure('gpu_actor_original_callback', existing)
            def direct():
                logits = actor.forward_logits(features, mask)
                if not bool(torch.isfinite(logits[mask]).all()):
                    raise ValueError('legal logits must be finite')
                probabilities = logits.softmax(-1).to(device)
                selected = torch.multinomial(probabilities, 1).squeeze(-1)
                selected.cpu().tolist()
                return selected
            measure('cpu_softmax_gpu_multinomial', direct)
            if args.prior_weight == 0:
                def numpy_hybrid():
                    distribution = Categorical(logits=numpy_logits().to(device))
                    selected = distribution.sample()
                    selected.cpu().tolist()
                    return selected
                measure('numpy_logits_gpu_categorical', numpy_hybrid)
            cpu_logits = actor.forward_logits(features, mask)
            gpu_logits = gpu_actor.forward_logits(features.to(device), mask.to(device)).cpu()
            result['legal_logit_max_abs_error'] = float((cpu_logits[mask] - gpu_logits[mask]).abs().max())
            result['probability_max_abs_error'] = float((cpu_logits.softmax(-1) - gpu_logits.softmax(-1)).abs().max())
            torch.cuda.manual_seed(3912)
            rng = torch.cuda.get_rng_state(device)
            reference = torch.stack([existing().cpu() for _ in range(200)])
            reference_rng = torch.cuda.get_rng_state(device)
            torch.cuda.set_rng_state(rng, device)
            trial = torch.stack([hybrid().cpu() for _ in range(200)])
            trial_rng = torch.cuda.get_rng_state(device)
            result['same_selected_actions'] = bool(torch.equal(reference, trial))
            result['same_cuda_rng_after'] = bool(torch.equal(reference_rng, trial_rng))
            result['sample_count'] = reference.numel()
            torch.cuda.set_rng_state(rng, device)
            direct_trial = torch.stack([direct().cpu() for _ in range(200)])
            result['direct_same_selected_actions'] = bool(torch.equal(reference, direct_trial))
            result['direct_action_mismatches'] = int((reference != direct_trial).sum())
            result['direct_same_cuda_rng_after'] = bool(torch.equal(reference_rng, torch.cuda.get_rng_state(device)))
            result['cpu_softmax_vs_gpu_categorical_probability_max_abs_error'] = float((cpu_logits.softmax(-1) - Categorical(logits=gpu_logits.to(device)).probs.cpu()).abs().max())
            if args.prior_weight == 0:
                torch.cuda.set_rng_state(rng, device)
                numpy_trial = torch.stack([numpy_hybrid().cpu() for _ in range(200)])
                result['numpy_same_selected_actions'] = bool(torch.equal(reference, numpy_trial))
                result['numpy_same_cuda_rng_after'] = bool(torch.equal(reference_rng, torch.cuda.get_rng_state(device)))
            result['reference_actions_sha256'] = hashlib.sha256(reference.numpy().tobytes()).hexdigest()
            with torch.enable_grad():
                actor.requires_grad_(True)
                gpu_actor.requires_grad_(True)
                for model, target_device in ((actor, torch.device('cpu')), (gpu_actor, device)):
                    distribution = Categorical(logits=model.forward_logits(features.to(target_device), mask.to(target_device)))
                    selected = torch.zeros(args.rows, dtype=torch.long, device=target_device)
                    advantage = torch.linspace(-2.5, 2.5, args.rows, device=target_device)
                    loss = -(distribution.log_prob(selected) * advantage + .01 * distribution.entropy()).mean()
                    loss.backward()
                result['parameter_gradient_max_abs_error'] = max(float((cpu.grad - gpu.grad.cpu()).abs().max()) for cpu, gpu in zip(actor.parameters(), gpu_actor.parameters()))
                result['cpu_gradient_max_abs'] = max(float(parameter.grad.abs().max()) for parameter in actor.parameters())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
