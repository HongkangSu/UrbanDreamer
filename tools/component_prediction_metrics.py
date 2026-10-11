"""Model-independent physical-unit ensemble diagnostics and clustered statistics.

Samples have shape (draws, states, roads, features), targets (states, roads,
features). Decode before calling: count is vehicles and speed is metres/second.
NaN targets mark unavailable labels, never zero-speed observations. All predicted
values must be finite. Metrics use the empirical predictive distribution (the
V-statistic, including diagonal sample pairs), including for deterministic models.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def empirical_crps(samples: Any, target: Any) -> np.ndarray:
    """Elementwise CRPS; sorting avoids allocating a draws-squared tensor."""
    samples, target = _array(samples), _array(target)
    if samples.ndim < 1 or not samples.shape[0] or samples.shape[1:] != target.shape:
        raise ValueError("samples must have shape (draws, *target.shape), draws > 0")
    if not np.isfinite(samples).all() or not np.isfinite(target).all():
        raise ValueError("CRPS inputs must be finite")
    draws = samples.shape[0]
    weights = (2 * np.arange(draws) - draws + 1).reshape(
        (draws,) + (1,) * target.ndim
    )
    half_pair_distance = (np.sort(samples, axis=0) * weights).sum(axis=0) / draws**2
    return np.maximum(np.abs(samples - target).mean(axis=0) - half_pair_distance, 0.)


def empirical_energy_score(samples: Any, target: Any) -> float:
    """Joint empirical energy score with Euclidean norm; input already scaled."""
    samples, target = _array(samples), _array(target)
    if (samples.ndim != 2 or target.ndim != 1 or samples.shape[1:] != target.shape
            or not samples.shape[0] or not target.size):
        raise ValueError("energy score requires (draws, dimensions) and (dimensions,)")
    if not np.isfinite(samples).all() or not np.isfinite(target).all():
        raise ValueError("energy score inputs must be finite")
    # One sample-pair row at a time: bounded memory even on large road graphs.
    pair_sum = sum(np.linalg.norm(samples - draw, axis=1).sum() for draw in samples)
    result = np.linalg.norm(samples - target, axis=1).mean() - pair_sum / (2 * len(samples)**2)
    return float(max(result, 0.))


def summarize_state_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Equal-state aggregation; pooled RMSE uses state MSE, not mean state RMSE.

    Metadata is deliberately excluded. For equal-demand/episode inference use
    grouped_bootstrap, rather than interpreting these descriptive means as CIs.
    """
    if not records:
        return {"states": 0}
    metric_keys = records[0].get("metric_keys")
    if metric_keys is None:
        raise ValueError("records require metric_keys from evaluate_prediction_samples")
    if any(record.get("metric_keys") != metric_keys for record in records):
        raise ValueError("cannot aggregate records with different metric protocols")
    summary: dict[str, Any] = {"states": len(records)}
    for key in metric_keys:
        values = [float(row[key]) for row in records if row.get(key) is not None]
        if not all(np.isfinite(values)):
            raise ValueError(f"nonfinite metric {key}")
        summary[key] = float(np.mean(values)) if values else None
        if "_rmse_" in key and values:
            summary[key] = float(np.sqrt(np.mean(np.square(values))))
    return summary


def evaluate_prediction_samples(
    samples: Any,
    target: Any,
    *,
    scales: Any,
    occupied_mask: Any | None = None,
    valid_mask: Any | None = None,
    state_metadata: Sequence[Mapping[str, Any]] | None = None,
    latent_samples: Any | None = None,
    latent_disagreement: Any | None = None,
    feature_names: Sequence[str] = ("count", "speed"),
    lower_bounds: Any | None = (0., 0.),
    upper_bounds: Any | None = None,
    cohort_count: Any | None = None,
    interval_levels: Sequence[float] = (.5, .8, .9, .95),
) -> dict[str, Any]:
    """Score physical-unit samples, preserving per-state records for later analysis.

    ``scales`` is a strictly positive fixed TRAINING-set feature scale, shape (D,).
    Optional bounds broadcast to (B,E,D), e.g. (E,D) for per-road speed limits.
    ``valid_mask`` is boolean (B,E,D); NaN targets are automatically unavailable.
    Occupied roads default to true count > 0, never predicted count > 0.
    The joint score uses every available coordinate; compare only identical masks
    and graphs, since the unnormalised Euclidean score depends on dimension.
    ``latent_disagreement`` can be the actual implementation's scalar per state;
    ``latent_samples`` additionally exposes population latent variance, explicitly
    named to avoid conflating different definitions of disagreement.
    """
    samples, target, scales = _array(samples), _array(target), _array(scales)
    if samples.ndim != 4 or target.shape != samples.shape[1:] or min(samples.shape) < 1:
        raise ValueError("expected samples (K,B,E,D) and target (B,E,D), all dimensions > 0")
    draws, states, roads, features = samples.shape
    if len(feature_names) != features or len(set(feature_names)) != features:
        raise ValueError("feature_names must be unique with one name per feature")
    if not np.isfinite(samples).all() or np.isinf(target).any():
        raise ValueError("samples must be finite; target may contain NaN but not infinity")
    if scales.shape != (features,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("scales must be positive finite fixed training feature scales (D,)")
    valid = np.isfinite(target)
    if valid_mask is not None:
        provided = np.asarray(valid_mask)
        if provided.shape != target.shape or provided.dtype != bool:
            raise ValueError("valid_mask must be boolean with target shape")
        valid &= provided
    if occupied_mask is None:
        if "count" not in feature_names:
            raise ValueError("supply occupied_mask when feature_names omits count")
        count_index = feature_names.index("count")
        occupied = valid[..., count_index] & (target[..., count_index] > 0)
    else:
        occupied = np.asarray(occupied_mask)
        if occupied.shape != (states, roads) or occupied.dtype != bool:
            raise ValueError("occupied_mask must be boolean (B,E)")
    metadata = [{} for _ in range(states)] if state_metadata is None else state_metadata
    if len(metadata) != states:
        raise ValueError("state_metadata must contain one mapping per state")
    for level in interval_levels:
        if not 0 < level < 1:
            raise ValueError("interval levels must be strictly between zero and one")
    if len({f"{100 * level:g}" for level in interval_levels}) != len(interval_levels):
        raise ValueError("interval levels must be unique")

    def bounds_array(value: Any | None) -> np.ndarray | None:
        if value is None:
            return None
        result = np.broadcast_to(_array(value), target.shape)
        if not np.isfinite(result).all():
            raise ValueError("bounds must be finite; use None to omit a bound")
        return result

    lower, upper = bounds_array(lower_bounds), bounds_array(upper_bounds)
    if lower is not None and upper is not None and np.any(lower > upper):
        raise ValueError("lower bounds must not exceed upper bounds")
    cohort = None if cohort_count is None else np.broadcast_to(_array(cohort_count), (states,))
    if cohort is not None and (not np.isfinite(cohort).all() or np.any(cohort < 0)):
        raise ValueError("cohort_count must be finite and nonnegative")
    latent = None if latent_samples is None else _array(latent_samples)
    if latent is not None and (latent.ndim < 3 or latent.shape[:2] != (draws, states)
                              or not np.isfinite(latent).all()):
        raise ValueError("latent_samples must be finite (K,B,...) with matching K,B")
    disagreement = None if latent_disagreement is None else _array(latent_disagreement)
    if disagreement is not None and (disagreement.shape != (states,)
                                    or not np.isfinite(disagreement).all()
                                    or np.any(disagreement < 0)):
        raise ValueError("latent_disagreement must be finite nonnegative (B,)")
    records = []
    for state in range(states):
        prediction, truth = samples[:, state], target[state]
        metrics: dict[str, Any] = {"draws": draws, "roads": roads,
                                  "occupied_roads": int(occupied[state].sum())}
        for feature, name in enumerate(feature_names):
            for scope, selection in (("all", valid[state, :, feature]),
                                     ("occupied", valid[state, :, feature] & occupied[state])):
                suffix = f"_{scope}"
                values, labels = prediction[:, selection, feature], truth[selection, feature]
                prefix = f"{name}_"
                metrics[prefix + "valid_roads" + suffix] = int(selection.sum())
                keys = ("mae", "mse", "rmse", "sample_mae", "crps", "sample_variance",
                        "below_lower_fraction", "above_upper_fraction")
                for key in keys:
                    metrics[prefix + key + suffix] = None
                for level in interval_levels:
                    metrics[f"{name}_coverage{100*level:g}{suffix}"] = None
                    metrics[f"{name}_width{100*level:g}{suffix}"] = None
                if not labels.size:
                    continue
                error = values.mean(axis=0) - labels
                metrics[prefix + "mae" + suffix] = float(np.abs(error).mean())
                metrics[prefix + "mse" + suffix] = float(np.square(error).mean())
                metrics[prefix + "rmse" + suffix] = float(np.sqrt(np.square(error).mean()))
                metrics[prefix + "sample_mae" + suffix] = float(np.abs(values - labels).mean())
                metrics[prefix + "crps" + suffix] = float(empirical_crps(values, labels).mean())
                metrics[prefix + "sample_variance" + suffix] = float(values.var(axis=0).mean())
                if lower is not None:
                    metrics[prefix + "below_lower_fraction" + suffix] = float(
                        (values < lower[state, selection, feature]).mean())
                if upper is not None:
                    metrics[prefix + "above_upper_fraction" + suffix] = float(
                        (values > upper[state, selection, feature]).mean())
                for level in interval_levels:
                    lo, hi = np.quantile(values, [(1-level)/2, (1+level)/2], axis=0)
                    metrics[f"{name}_coverage{100*level:g}{suffix}"] = float(
                        ((labels >= lo) & (labels <= hi)).mean())
                    metrics[f"{name}_width{100*level:g}{suffix}"] = float((hi - lo).mean())
        selection = valid[state]
        metrics["energy_score_dimensions"] = int(selection.sum())
        metrics["energy_score"] = (empirical_energy_score(
            (prediction / scales)[:, selection], (truth / scales)[selection]
        ) if selection.any() else None)
        count_keys = ("true_total_count", "predicted_total_count", "total_count_bias",
                      "total_count_absolute_error", "total_count_sample_absolute_error",
                      "total_count_sample_variance", "total_count_over_cohort_fraction",
                      "total_count_max_cohort_excess")
        metrics.update(dict.fromkeys(count_keys))
        if "count" in feature_names:
            index = feature_names.index("count")
            # A partial sum is not a total. Missing count labels make mass metrics unavailable.
            if valid[state, :, index].all():
                totals = prediction[..., index].sum(axis=-1)
                true_total = float(truth[:, index].sum())
                bias = float(totals.mean() - true_total)
                metrics.update(true_total_count=true_total, predicted_total_count=float(totals.mean()),
                               total_count_bias=bias, total_count_absolute_error=abs(bias),
                               total_count_sample_absolute_error=float(np.abs(totals-true_total).mean()),
                               total_count_sample_variance=float(totals.var()))
                if cohort is not None:
                    metrics["total_count_over_cohort_fraction"] = float(
                        (totals > cohort[state] + 1e-6).mean())
                    metrics["total_count_max_cohort_excess"] = float(
                        max(0., totals.max() - cohort[state]))
        if latent is not None:
            metrics["latent_sample_variance"] = float(latent[:, state].var(axis=0).mean())
        if disagreement is not None:
            metrics["latent_disagreement"] = float(disagreement[state])
        reserved = set(metrics) | {"metric_keys"}
        if reserved.intersection(metadata[state]):
            raise ValueError(f"metadata collides with metrics: {sorted(reserved.intersection(metadata[state]))}")
        records.append({**dict(metadata[state]), **metrics, "metric_keys": list(metrics)})
    return {
        "summary": summarize_state_records(records),
        "states": records,
        "protocol": {
            "features": list(feature_names), "count_unit": "vehicles", "speed_unit": "m/s",
            "scales": scales.tolist(), "scales_source": "caller_supplied_fixed_training_data",
            "aggregation": "equal state; within state equal valid roads; RMSE from state MSE",
            "crps": "empirical V-statistic including diagonal pairs",
            "energy_score": "Euclidean norm of available scaled road-feature vector; V-statistic",
            "interval_levels": list(interval_levels), "interval_quantile_method": "linear",
            "variance_ddof": 0, "missing_target": "NaN excluded by feature",
            "warning": "use identical target masks across models; no independent-window confidence intervals",
        },
    }


def grouped_bootstrap(
    records: Sequence[Mapping[str, Any]], metric: str, *,
    group_keys: Sequence[str] = ("demand_group", "episode"),
    n_bootstrap: int = 1000, confidence: float = .95, seed: int = 0,
) -> dict[str, Any]:
    """Hierarchical cluster bootstrap, retaining all windows within each episode.

    The estimand gives equal weight to demands, then episodes, then windows.
    Each level is resampled with replacement; windows themselves are NOT sampled.
    Filter variant/horizon first. Use an archive-unique episode key if episode
    numbers are reused across methods. A single top-level cluster yields no CI.
    """
    if not group_keys or n_bootstrap < 1 or not 0 < confidence < 1:
        raise ValueError("need group_keys, positive n_bootstrap, confidence in (0,1)")
    tree: dict[Any, Any] = {}
    valid_rows = 0
    for row in records:
        value = row.get(metric)
        if value is None:
            continue
        if not np.isfinite(value):
            raise ValueError(f"nonfinite {metric}")
        if any(key not in row or row[key] is None for key in group_keys):
            raise ValueError(f"all valid records require grouping keys {tuple(group_keys)}")
        node = tree
        for key in group_keys[:-1]:
            node = node.setdefault(row[key], {})
        node.setdefault(row[group_keys[-1]], []).append(float(value))
        valid_rows += 1
    rng = np.random.default_rng(seed)

    def statistic(node: Any, resample: bool) -> float:
        if isinstance(node, list):
            # RMSE is nonlinear: aggregate squared state errors before root.
            return float(np.mean(np.square(node) if "_rmse_" in metric else node))
        children = list(node.values())
        indices = rng.integers(len(children), size=len(children)) if resample else range(len(children))
        return float(np.mean([statistic(children[i], resample) for i in indices]))

    def transform(value: float) -> float:
        return float(np.sqrt(value)) if "_rmse_" in metric else value

    result: dict[str, Any] = {
        "metric": metric, "group_keys": list(group_keys), "valid_states": valid_rows,
        "top_level_groups": len(tree), "estimate": None, "ci_low": None, "ci_high": None,
        "confidence": confidence, "n_bootstrap": n_bootstrap, "seed": seed,
        "estimand": "equal weight recursively across groups, then all windows within episode",
    }
    if not tree:
        return result
    result["estimate"] = transform(statistic(tree, False))
    if len(tree) >= 2:
        distribution = [transform(statistic(tree, True)) for _ in range(n_bootstrap)]
        result["ci_low"], result["ci_high"] = map(float, np.quantile(
            distribution, [(1-confidence)/2, (1+confidence)/2]))
    return result


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ordered = values[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + end - 1) / 2
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(_ranks(x), _ranks(y))[0, 1])


def disagreement_analysis(
    records: Sequence[Mapping[str, Any]], *,
    disagreement_key: str = "latent_disagreement", error_key: str = "count_mae_all",
    congestion_key: str = "congestion", group_keys: Sequence[str] = ("demand_group", "episode"),
    congestion_edges: Sequence[float] | None = None,
    n_bootstrap: int = 300, seed: int = 0,
) -> dict[str, Any]:
    """Descriptive window Spearman, grouped diagnostics, and risk-decile plot data.

    Spearman has no IID-window p-value. Confidence intervals are instead supplied
    for risk-bin means with the hierarchical bootstrap. Equal-disagreement ties
    stay together; consequently degenerate predictions may occupy just one bin.
    Congestion edges should be fixed from training data when comparing models.
    """
    if not group_keys:
        raise ValueError("group_keys must not be empty")
    valid = []
    for row in records:
        values = [row.get(key) for key in (disagreement_key, error_key)]
        if any(value is None for value in values):
            continue
        if not np.isfinite(values).all():
            raise ValueError("nonfinite disagreement or error")
        if any(key not in row or row[key] is None for key in group_keys):
            raise ValueError("disagreement analysis requires demand/episode group identifiers")
        valid.append(row)

    def analyze(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        x = np.asarray([row[disagreement_key] for row in rows], dtype=float)
        y = np.asarray([row[error_key] for row in rows], dtype=float)
        clusters: dict[tuple, list] = defaultdict(list)
        for row in rows:
            clusters[tuple(row[key] for key in group_keys)].append(row)
        cluster_x = np.asarray([np.mean([row[disagreement_key] for row in group])
                                for group in clusters.values()])
        cluster_y = np.asarray([np.mean([row[error_key] for row in group])
                                for group in clusters.values()])
        deciles = []
        if len(rows):
            edges = np.unique(np.quantile(x, np.arange(1, 10) / 10))
            bins = np.searchsorted(edges, x, side="left")
            for bin_id in np.unique(bins):
                chosen = [row for row, index in zip(rows, bins) if index == bin_id]
                estimate = grouped_bootstrap(chosen, error_key, group_keys=group_keys,
                                             n_bootstrap=n_bootstrap, seed=seed + int(bin_id))
                deciles.append({"bin": int(bin_id) + 1, "states": len(chosen),
                    "disagreement_min": float(min(row[disagreement_key] for row in chosen)),
                    "disagreement_max": float(max(row[disagreement_key] for row in chosen)),
                    "disagreement_mean": grouped_bootstrap(
                        chosen, disagreement_key, group_keys=group_keys, n_bootstrap=1, seed=seed)["estimate"],
                    "error": estimate})
        return {"states": len(rows), "episodes": len(clusters),
                "spearman_descriptive_windows": _spearman(x, y),
                "spearman_episode_means": _spearman(cluster_x, cluster_y), "deciles": deciles}

    result = {"all": analyze(valid), "congestion_strata": [],
              "protocol": {"disagreement_key": disagreement_key, "error_key": error_key,
                           "spearman": "descriptive only; no IID window p-values",
                           "risk_mean_ci": "hierarchical demand/episode cluster bootstrap",
                           "tie_handling": "identical disagreements never split between bins"}}
    with_congestion = [row for row in valid if row.get(congestion_key) is not None]
    if any(not np.isfinite(row[congestion_key]) for row in with_congestion):
        raise ValueError("congestion must be finite when supplied")
    if with_congestion:
        values = np.asarray([row[congestion_key] for row in with_congestion])
        edges = (np.unique(np.quantile(values, [1/3, 2/3])) if congestion_edges is None
                 else _array(congestion_edges))
        if edges.ndim != 1 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
            raise ValueError("congestion_edges must be finite, strictly increasing")
        result["protocol"]["congestion_edges"] = edges.tolist()
        result["protocol"]["congestion_edges_source"] = (
            "diagnostic evaluation quantiles" if congestion_edges is None else "caller_supplied")
        bins = np.searchsorted(edges, values, side="left")
        for index in np.unique(bins):
            chosen = [row for row, bin_id in zip(with_congestion, bins) if bin_id == index]
            result["congestion_strata"].append({"stratum": int(index), **analyze(chosen)})
    return result
