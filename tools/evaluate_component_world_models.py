"""Development-only paired physical prediction diagnostics for frozen WM variants."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tools.component_experiment_data import atomic_json, dataset, load_frozen, sha256
from tools.component_prediction_metrics import (
    disagreement_analysis, evaluate_prediction_samples, grouped_bootstrap, summarize_state_records,
)
from tools.world_model_evaluation import _repeat_events
from urban_dreamer.learning.cohort_targets import prepare_cohort_batch, prepare_transition_targets
from urban_dreamer.models.experimental_dynamics import apply_variant, variant_metadata


def stable_seed(*values):
    return int(hashlib.sha256(":".join(map(str, values)).encode()).hexdigest()[:8], 16)


def select_conditions(entries, *, per_group, max_groups, horizon, seed):
    """Choose starts with every requested physical window, independent of model.

    The CLI requests max forecast horizon + one following window for speed
    labels, so every selected state belongs to the full-horizon intersection.
    """
    if min(per_group, max_groups, horizon) < 1:
        raise ValueError("condition/group/horizon counts must be positive")
    groups = defaultdict(list)
    for index, entry in enumerate(entries):
        if entry["windows"] >= horizon:
            groups[entry["demand_group"]].append((index, entry))
    selected = []
    for group in sorted(groups, key=lambda x: stable_seed(seed, x))[:max_groups]:
        episodes = sorted(groups[group], key=lambda item: stable_seed(seed, item[1]["path"], item[1]["episode"]))
        seen = set()
        for condition in range(per_group):
            index, entry = episodes[condition % len(episodes)]
            legal = entry["windows"] - horizon + 1
            rng = np.random.default_rng(stable_seed(seed, group, condition))
            starts = rng.permutation(legal)
            start = next((int(value) for value in starts if (index, int(value)) not in seen), None)
            if start is None:
                continue
            seen.add((index, start))
            selected.append(dict(episode_index=index, start=start, demand_group=group,
                episode=f'{entry["path"]}::episode={entry["episode"]}', source_path=entry["path"],
                requested_horizon=horizon, condition_seed=stable_seed(seed, group, entry["path"], entry["episode"], start)))
    if not selected:
        raise ValueError(f"no development conditions have all {horizon} required windows")
    return selected


def select_traffic_conditions(data, capacities, *, per_group, max_groups, horizon, seed, include_empty=True):
    """Predeclared observed-count phases; selection never consults predictions.

    Per archive: early is the first pre-peak >=25% of the eligible peak;
    peak is the maximum; drain is the first post-peak <=50% of the peak.
    Without a drain crossing use the final eligible post-peak boundary and label
    it explicitly. Zero-count starts form a separate stress stratum. Both use
    only archives/starts with all ``horizon`` required physical windows.
    """
    entries = data.manifest["entries"]
    seeds = select_conditions(entries, per_group=1, max_groups=max_groups, horizon=horizon, seed=seed)
    selected = []
    for seed_row in seeds:
        group = seed_row["demand_group"]
        episodes = sorted([(i, e) for i, e in enumerate(entries)
                           if e["demand_group"] == group and e["windows"] >= horizon],
            key=lambda item: stable_seed(seed, item[1]["path"], item[1]["episode"]))
        seen = set()
        for condition in range(per_group):
            index, entry = episodes[(condition // 3) % len(episodes)]
            arrays = data._episode(index, flow_only=True)
            legal = int(entry["windows"]) - horizon + 1
            counts = (np.asarray(arrays["flow"][:legal, :, 2], dtype=np.float64)
                      * capacities.cpu().numpy()).sum(-1)
            peak = int(np.argmax(counts))
            if counts[peak] <= 1e-6:
                continue
            early = int(np.flatnonzero(counts[:peak + 1] >= .25 * counts[peak])[0])
            after = np.flatnonzero(counts[peak + 1:] <= .5 * counts[peak])
            drain = peak + 1 + int(after[0]) if len(after) else len(counts) - 1
            phase = ("early", "peak", "drain" if len(after) else "post_peak_no_drain")[condition % 3]
            start = (early, peak, drain)[condition % 3]
            if (index, start) in seen:
                continue
            seen.add((index, start))
            selected.append(dict(episode_index=index, start=start, demand_group=group,
                episode=f'{entry["path"]}::episode={entry["episode"]}', source_path=entry["path"],
                requested_horizon=horizon, traffic_phase=phase, initial_total_count=float(counts[start]),
                condition_seed=stable_seed(seed, group, entry["path"], entry["episode"], start)))
        if include_empty:
            for index, entry in episodes:
                arrays = data._episode(index, flow_only=True)
                legal = int(entry["windows"]) - horizon + 1
                counts = (np.asarray(arrays["flow"][:legal, :, 2], dtype=np.float64)
                          * capacities.cpu().numpy()).sum(-1)
                empty = np.flatnonzero(counts <= 1e-6)
                if not len(empty):
                    continue
                start = int(empty[-1])
                selected.append(dict(episode_index=index, start=start, demand_group=group,
                    episode=f'{entry["path"]}::episode={entry["episode"]}', source_path=entry["path"],
                    requested_horizon=horizon, traffic_phase="empty_stress", initial_total_count=float(counts[start]),
                    condition_seed=stable_seed(seed, group, entry["path"], entry["episode"], start)))
                break
    if not selected:
        raise ValueError("no traffic-phase conditions available")
    return selected


def physical(values, capacities, limits):
    """Map normalized decoder/traffic coordinates to count and m/s, un-clipped."""
    return torch.stack((values[..., 2] * capacities.to(values),
                        values[..., 1] * limits.to(values)), dim=-1)


def endpoint_target(endpoint, following, capacities, limits):
    target = physical(endpoint, capacities, limits)
    target[..., 1] = (float("nan") if following is None else
                     .5 * (endpoint[..., 1] + following[..., 1]) * limits.to(endpoint))
    return target


@torch.no_grad()
def sample_trajectories(model, initial, action, events, *, horizons, samples, sample_chunk,
                        steps, capacities, limits, seed):
    """Maintain independent latent trajectories; retain physical samples on CPU."""
    if samples < 2 or sample_chunk < 1:
        raise ValueError("need at least two samples and a positive sample chunk")
    torch.manual_seed(seed)
    physical_draws = {h: [] for h in horizons}
    moments = {h: [None, None, 0] for h in horizons}
    for begin in range(0, samples, sample_chunk):
        count = min(sample_chunk, samples - begin)
        current = initial.repeat(count, 1, 1)
        for step in range(max(horizons)):
            current = model.transition(current, action[:, step].repeat(count, 1),
                action_events=_repeat_events(events.for_step(step), count), steps=steps, solver="euler")
            h = step + 1
            if h not in physical_draws:
                continue
            decoded = physical(model.decode(current), capacities, limits)
            physical_draws[h].append(decoded.reshape(count, initial.shape[0], *decoded.shape[1:]).cpu())
            latent = current.reshape(count, initial.shape[0], *current.shape[1:]).double().cpu()
            # Parallel Welford avoids cancellation for nearly deterministic predictors.
            mean, m2, n = moments[h]
            new_mean, new_m2 = latent.mean(0), ((latent - latent.mean(0)) ** 2).sum(0)
            if n:
                delta = new_mean - mean
                new_m2 = m2 + new_m2 + delta.square() * n * count / (n + count)
                new_mean = mean + delta * count / (n + count)
            moments[h] = [new_mean, new_m2, n + count]
    return ({h: torch.cat(draws) for h, draws in physical_draws.items()},
            {h: (m2 / (n - 1)).mean(dim=(-1, -2)).numpy() for h, (_, m2, n) in moments.items()})


@torch.no_grad()
def evaluate_condition(model, batch, capacities, limits, *, horizons, samples, sample_chunk,
                       steps, device, scales, metadata, seed):
    flow = batch.flow.to(device)
    capacities, limits = capacities.to(device), limits.to(device)
    frames = flow.shape[1]
    available = [h for h in horizons if h < frames]
    if not available:
        raise ValueError("condition lacks requested forecast endpoints")
    one_step = prepare_transition_targets(batch, capacities=capacities.cpu())
    cohort = prepare_cohort_batch(batch, capacities=capacities.cpu())
    protocols = [
        ("one_step_previous_state", [1], flow[:, 0], batch.action.to(device), batch.action_events.to(device),
         one_step.transition_target_flow.to(device), one_step.transition_cohort_count[:, 0], False),
        ("recorded_action_recursive_initial_cohort", available, cohort.transition_flow[:, 0].to(device),
         cohort.transition_action.to(device), cohort.transition_action_events.to(device),
         cohort.transition_flow.to(device), cohort.cohort_count, True),
    ]
    rows = []
    occupied = flow[:, 0, :, 2] > 0
    congestion = ((1 - flow[:, 0, :, 1]).clamp_min(0) * occupied).sum(-1) / occupied.sum(-1).clamp_min(1)
    for protocol, forecast_horizons, start, action, events, targets, cohort_count, recursive in protocols:
        initial = model.encode(start)
        draws, disagreement = sample_trajectories(model, initial, action, events,
            horizons=forecast_horizons, samples=samples, sample_chunk=sample_chunk, steps=steps,
            capacities=capacities, limits=limits, seed=stable_seed(seed, protocol, "mean"))
        if 1 in forecast_horizons:
            _, independent_disagreement = sample_trajectories(model, initial, action, events,
                horizons=[1], samples=samples, sample_chunk=sample_chunk, steps=steps,
                capacities=capacities, limits=limits, seed=stable_seed(seed, protocol, "risk"))
            disagreement[1] = independent_disagreement[1]
        for h in forecast_horizons:
            endpoint = targets[:, h if recursive else 0]
            following = flow[:, h + 1] if h + 1 < frames else None
            target = endpoint_target(endpoint, following, capacities, limits).cpu()
            baselines = {"predictor": draws[h],
                "persistence": physical(start, capacities, limits).cpu().unsqueeze(0),
                "endpoint_codec": physical(model.decode(model.encode(endpoint)), capacities, limits).cpu().unsqueeze(0)}
            for predictor, values in baselines.items():
                details = dict(metadata, protocol=protocol, predictor=predictor, horizon=h,
                    forecast_seconds=30 * h, congestion=float(congestion[0]),
                    disagreement_independent_samples=(h == 1 and predictor == "predictor"))
                metrics = evaluate_prediction_samples(values, target, scales=scales,
                    cohort_count=cohort_count.cpu().numpy(), state_metadata=[details],
                    latent_disagreement=disagreement[h] if predictor == "predictor" else np.zeros(1),
                    upper_bounds=torch.stack((torch.full_like(limits, float(cohort_count[0])), limits), -1).cpu())
                rows.extend(metrics["states"])
    return rows


def estimate_training_scales(training, capacities, limits, *, seed, max_groups=32):
    """Fixed empirical population std from deterministic TRAINING start boundaries."""
    selected = select_conditions(training.manifest["entries"], per_group=2,
        max_groups=max_groups, horizon=1, seed=seed)
    total, squared, n = np.zeros(2), np.zeros(2), 0
    congestion = []
    for item in selected:
        arrays = training._episode(item["episode_index"], flow_only=True)
        frame = torch.as_tensor(np.asarray(arrays["flow"][item["start"]]).copy())
        values = physical(frame, capacities.cpu(), limits.cpu()).double().numpy()
        total += values.sum(0)
        squared += np.square(values).sum(0)
        n += len(values)
        occupied = frame[:, 2] > 0
        congestion.append(float((1 - frame[occupied, 1]).clamp_min(0).mean()) if occupied.any() else 0.)
    scales = np.sqrt(np.maximum(squared / n - np.square(total / n), 0)).clip(1e-6)
    edges = np.unique(np.quantile(congestion, [1/3, 2/3])).tolist()
    return scales, {"source": "deterministic_training_start_boundaries", "conditions": selected,
                    "road_observations": n, "scales": scales.tolist(), "congestion_edges": edges}


def summarize(records, *, seed):
    groups = defaultdict(list)
    for row in records:
        phase = row.get("traffic_phase", "uniform")
        strata = [phase] if phase in {"empty_stress", "uniform"} else ["traffic_phases"]
        if phase not in {"empty_stress", "uniform"}:
            strata.append(phase)
        for stratum in strata:
            groups[(row["protocol"], row["predictor"], row["horizon"], stratum)].append(row)
    summaries = []
    for (protocol, predictor, horizon, stratum), rows in groups.items():
        selected_metrics = ("count_mae_all", "count_mae_occupied", "speed_mae_occupied",
            "count_crps_occupied", "speed_crps_occupied", "count_coverage90_occupied",
            "speed_coverage90_occupied", "energy_score", "total_count_absolute_error")
        summaries.append(dict(protocol=protocol, predictor=predictor, horizon=horizon, stratum=stratum,
            descriptive=summarize_state_records(rows),
            clustered={key: grouped_bootstrap(rows, key, seed=seed) for key in selected_metrics}))
    return summaries


def run(args):
    if any((args.output_dir / name).exists() for name in ("protocol.json", "states.jsonl", "summary.json")):
        raise ValueError("evaluation output already exists; preserve it and choose a new output directory")
    if min(args.samples, args.sample_chunk, args.conditions_per_group, args.max_groups, args.steps) < 1 or args.samples < 2:
        raise ValueError("positive budgets and at least two samples are required")
    horizons = sorted(set(int(value) for value in args.horizons.split(",")))
    if not horizons or min(horizons) < 1:
        raise ValueError("horizons must contain positive integers")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        cap = min(args.gpu_memory_fraction, (2 * 1024**3) / total)
        if free < 2 * 1024**3:
            raise RuntimeError("GPU admission requires at least 2 GiB currently free")
        torch.cuda.set_per_process_memory_fraction(cap, device)
    checkpoint = None if args.checkpoint is None else torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config, prepared, topology, model, protocol = load_frozen(args.experiment_dir, device="cpu",
        fresh_dynamics_seed=None if checkpoint is None else checkpoint["seed"])
    if checkpoint is not None:
        if checkpoint.get("schema_version") != 1:
            raise ValueError("unsupported component checkpoint schema")
        checkpoint_protocol = checkpoint["protocol"].get("parent", checkpoint["protocol"])
        if any(checkpoint_protocol[key] != protocol[key] for key in
               ("development_manifest_identity", "checkpoint_sha256", "config_sha256")):
            raise ValueError("checkpoint belongs to a different data protocol")
        apply_variant(model, checkpoint["variant"], checkpoint["seed"])
        model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    validation = dataset(args.experiment_dir, config, prepared, topology, split="validation", horizon=max(horizons) + 1)
    training = dataset(args.experiment_dir, config, prepared, topology, split="train", horizon=1)
    capacities = torch.as_tensor(topology.network.capacities, dtype=torch.float32).clamp_min(1)
    limits = torch.as_tensor(topology.network.speed_limits_mps, dtype=torch.float32)
    selection_options = dict(per_group=args.conditions_per_group, max_groups=args.max_groups,
                             horizon=max(horizons) + 1, seed=protocol["selection_seed"])
    conditions = (select_conditions(validation.manifest["entries"], **selection_options)
        if args.condition_sampling == "uniform" else select_traffic_conditions(validation, capacities,
            **selection_options, include_empty=args.include_empty_stress))
    scales, scale_protocol = estimate_training_scales(training, capacities, limits, seed=protocol["selection_seed"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "conditions.json", conditions)
    evaluation_protocol = dict(schema_version=1, role="development_not_final_test", campaign=protocol,
        checkpoint_sha256=protocol["checkpoint_sha256"] if args.checkpoint is None else sha256(args.checkpoint),
        training_protocol=None if checkpoint is None else checkpoint["protocol"],
        training_update=None if checkpoint is None else checkpoint["update"],
        model=variant_metadata(model), horizons=horizons, samples=args.samples, sample_chunk=args.sample_chunk,
        steps=args.steps, solver="euler", scales=scale_protocol,
        condition_sampling=args.condition_sampling, empty_stress_aggregated_separately=True,
        phase_definition="early first >=25% eligible peak; peak argmax; drain first postpeak <=50%; otherwise final eligible boundary labelled post_peak_no_drain",
        target_speed="mean endpoint and following observed boundary; unavailable labels are NaN",
        evaluation_population="complete-horizon intersection including following speed-label window",
        required_physical_windows=max(horizons) + 1,
        short_episode_policy="skip episodes shorter than required_physical_windows before group selection",
        missing_horizons="CLI rejects incomplete batches; direct evaluate_condition supports missing-label diagnostics",
        disagreement="h1 separate noise ensemble; h>1 same-ensemble diagnostic only",
        limits="recorded conditions with background traffic; not a SUMO counterfactual or closed-loop policy validation")
    atomic_json(args.output_dir / "protocol.json", evaluation_protocol)
    records = []
    with (args.output_dir / "states.jsonl").open("w") as stream:
        for index, item in enumerate(conditions):
            batch = validation.batch_from_starts(item["episode_index"], [item["start"]], fm_only=False)
            if batch.flow.shape[1] < max(horizons) + 2:
                raise ValueError("selected condition lacks the complete forecast horizon and following speed-label boundary")
            rows = evaluate_condition(model, batch, capacities, limits, horizons=horizons, samples=args.samples,
                sample_chunk=args.sample_chunk, steps=args.steps, device=device, scales=scales,
                metadata=dict(item, variant=variant_metadata(model)["variant"]), seed=item["condition_seed"])
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            records.extend(rows)
            print(json.dumps({"conditions_done": index + 1, "conditions_total": len(conditions)}), flush=True)
    risk = [row for row in records if row["predictor"] == "predictor" and row["horizon"] == 1]
    risk_by_protocol = {}
    for name in sorted({row["protocol"] for row in risk}):
        selected = [row for row in risk if row["protocol"] == name and row.get("traffic_phase") != "empty_stress"]
        risk_by_protocol[name] = {error: disagreement_analysis(selected, error_key=error,
            congestion_edges=scale_protocol["congestion_edges"], seed=protocol["selection_seed"])
            for error in ("count_mae_occupied", "speed_mae_occupied", "total_count_absolute_error")}
    result = dict(status="complete", role="development_not_final_test", conditions=len(conditions),
                  evaluations=summarize(records, seed=protocol["selection_seed"]), disagreement=risk_by_protocol)
    atomic_json(args.output_dir / "summary.json", result)
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--experiment-dir", type=Path, default=Path("outputs/component_experiments_20261011"))
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--device", default="cpu")
    result.add_argument("--samples", type=int, default=16)
    result.add_argument("--sample-chunk", type=int, default=1)
    result.add_argument("--conditions-per-group", type=int, default=3)
    result.add_argument("--condition-sampling", choices=("uniform", "traffic_phase"), default="traffic_phase")
    result.add_argument("--include-empty-stress", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--max-groups", type=int, default=32)
    result.add_argument("--horizons", default="1,2,4,8,16")
    result.add_argument("--steps", type=int, default=2)
    result.add_argument("--threads", type=int, default=2)
    result.add_argument("--gpu-memory-fraction", type=float, default=.06)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
