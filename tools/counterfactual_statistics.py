"""Paired effect summaries with uncertainty over independent experimental units.

For routing experiments, repeats of a deterministic replay of the same state are
reproducibility checks, not independent observations. Use
``cluster_paired_summary`` to average effects within each demand seed before
estimating uncertainty across seeds. The percentile bootstrap is descriptive and
can be unreliable with very few independent units.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


_BOOTSTRAP_DRAWS = 20_000
_BOOTSTRAP_SEED = 0


def paired_summary(differences: Iterable[float], confidence: float = .95) -> dict:
    """Summarize signed paired differences using a reproducible bootstrap.

    Positive and negative counts retain the caller's sign convention. The signal
    to noise ratio is ``abs(mean) / sample_sd``; it is not a significance test.
    Undefined statistics are ``None`` so the result serializes as strict JSON.
    A zero-variance bootstrap has a point interval and an explicit warning; it
    cannot establish uncertainty beyond the observed experimental units.
    """
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be finite and strictly between 0 and 1")
    values = np.asarray(list(differences), dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("differences must contain only finite scalar values")

    n = len(values)
    mean = float(values.mean()) if n else None
    sd = float(values.std(ddof=1)) if n > 1 else None
    se = sd / np.sqrt(n) if sd is not None else None
    result = {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": float(se) if se is not None else None,
        "ci_low": None,
        "ci_high": None,
        "confidence": float(confidence),
        "positive_count": int(np.count_nonzero(values > 0)),
        "negative_count": int(np.count_nonzero(values < 0)),
        "zero_count": int(np.count_nonzero(values == 0)),
        "signal_to_noise": abs(mean) / sd if sd is not None and sd > 0 else None,
        "uncertainty_method": "insufficient_independent_units",
        "bootstrap_draws": 0,
        "bootstrap_seed": _BOOTSTRAP_SEED,
        "variance_degenerate": sd == 0 if sd is not None else False,
        "uncertainty_note": "At least two independent units are needed for an interval.",
    }
    if n < 2:
        return result
    result.update(uncertainty_method="paired_percentile_bootstrap",
                  bootstrap_draws=_BOOTSTRAP_DRAWS)
    if sd == 0:
        result.update(ci_low=mean, ci_high=mean,
                      uncertainty_note="Zero observed variance gives a degenerate bootstrap interval; "
                                       "identical deterministic replays are not independent evidence.")
        return result

    # Bound temporary allocation while preserving a deterministic random stream.
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    batch_size = max(1, min(_BOOTSTRAP_DRAWS, 1_000_000 // n))
    bootstrap_means = np.empty(_BOOTSTRAP_DRAWS, dtype=np.float64)
    for start in range(0, _BOOTSTRAP_DRAWS, batch_size):
        end = min(start + batch_size, _BOOTSTRAP_DRAWS)
        indices = rng.integers(0, n, size=(end - start, n))
        bootstrap_means[start:end] = values[indices].mean(axis=1)
    tail = (1 - confidence) / 2
    low, high = np.quantile(bootstrap_means, [tail, 1 - tail])
    result.update(ci_low=float(low), ci_high=float(high),
                  uncertainty_note="Resample independent paired units; intervals with few units "
                                   "are descriptive and may have poor coverage.")
    return result


def cluster_paired_summary(rows: Iterable[Mapping[str, Any]],
                           cluster_key: str = "demand_seed",
                           value_key: str = "difference",
                           confidence: float = .95) -> dict:
    """Average each cluster's paired effects, then give clusters equal weight.

    This prevents extra states or deterministic replays under one demand seed
    from inflating the sample size. ``n`` is the number of independent clusters;
    ``n_pairs`` records all provided rows. Missing keys raise ``KeyError``.
    """
    clusters = defaultdict(list)
    n_pairs = 0
    for row in rows:
        value = float(row[value_key])
        if not np.isfinite(value):
            raise ValueError("differences must contain only finite scalar values")
        clusters[row[cluster_key]].append(value)
        n_pairs += 1
    cluster_means = [float(np.mean(values)) for values in clusters.values()]
    result = paired_summary(cluster_means, confidence=confidence)
    result.update(n_pairs=n_pairs, n_clusters=len(clusters),
                  cluster_key=cluster_key, value_key=value_key,
                  aggregation="equal_weight_cluster_means")
    return result
