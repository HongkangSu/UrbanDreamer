## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-07-23
- Verification Status: VERIFIED
- Version Label: hdv_od_route_validation_v1

## Experiment Result

- **ID**: `hdv_od_route_predictor_offline_20260723`
- **Type**: offline training and chronological one-step validation
- **Status**: completed
- **Working Directory**: `/home/wangwenrui/shk/UrbanDreamer`
- **Primary Metric**: projected edge total-variation distance versus persistence
- **Random Seed**: 42

### Question

Does an HDV predictor that first models each OD's next route distribution,
using the previous RouteRL probability/cost state and realized route travel
times, produce a more faithful episode-level HDV response than the deployed
edge-level MLP?

### Causal Prediction Protocol

For target episode \(e\), the structured model uses only information available
after episode \(e-1\):

1. the RouteRL probability and cost state used in \(e-1\);
2. realized per-OD-route travel times from \(e-1\), pooled over completed CAV
   and HDV trips;
3. the observed-route mask, previous CAV OD-route distribution, HDV OD
   population weight, and episode progress.

The existing `gawron_step_from_route_tt` transition produces the next
OD-route distribution. A shared neural residual is trained on top of that
anchor. Its epoch and residual scale are selected on a chronological
validation block. The residual is retained only when it improves validation
edge TV over the anchor by at least \(10^{-4}\); scale zero means the exact
RouteRL anchor is retained. OD-route probabilities are projected to edges with
HDV population weighting.

The time splits are:

| Network | Train | Validation | Test |
|---|---:|---:|---:|
| Gretz-Armainvilliers | 151--250 | 251--300 | 301--450 |
| Ingolstadt | 151--250 | 251--300 | 301--450 |
| Fontenay-en-Parisis | 1--50 | 51--100 | 101--150 |

### Main Results

All TV values below are multiplied by \(10^3\). The confidence interval is
for the paired difference `OD-route - persistence`, estimated with 2,000
moving-block bootstrap replicates and block length ten.

| Network | Persistence | Legacy edge MLP | OD-route | Delta [95% CI] | Active-OD route TV | Route top-1 | Direction cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gretz-Armainvilliers | 4.171 | 10.505 | 6.516 | +2.345 [1.899, 2.879] | 6.319 | 99.389% | 0.419 |
| Ingolstadt | 0.183 | 2.223 | 0.115 | -0.068 [-0.125, -0.016] | 1.692 | 99.837% | 0.569 |
| Fontenay-en-Parisis | 3.773 | 23.065 | 4.022 | +0.248 [-0.105, 0.531] | 2.602 | 99.774% | 0.586 |

The structured route transition is more accurate than the deployed edge MLP
on all three networks. It beats persistence only on Ingolstadt with a paired
interval entirely below zero. Gretz has a response-magnitude overshoot, and
Fontenay is statistically inconclusive with a slightly worse mean than
persistence. The learned neural residual did not pass the validation margin
on any network, so all saved route checkpoints record residual scale zero.
The experiment therefore validates the structured route-to-edge
reconstruction, not an additional learned human-adaptation residual.

### Output Files

| File | Purpose |
|---|---|
| `gretz/summary.json` | Full split, training, route, edge, response, and bootstrap metrics |
| `ingolstadt/summary.json` | Full split, training, route, edge, response, and bootstrap metrics |
| `fontenay/summary.json` | Full split, training, route, edge, response, and bootstrap metrics |
| `*/per_episode_metrics.csv` | Paired episode-level errors and response diagnostics |
| `*/residual_od_route_predictor.pt` | Route model checkpoint; selected residual scale is zero |
| `aggregate_summary.csv` | Cross-network paper-table values |

### Reproducibility

- **Method**: re-run stochastic/environment-sensitive numerical training with
  the same data, split, seed, and command.
- **Verdict**: REPRODUCIBLE.
- Gretz and Ingolstadt were rerun with the final sparse projection
  implementation and matched the earlier dense-projection metrics to the
  displayed precision.
- Fontenay was rerun into
  `/tmp/hdv_od_route_fontenay_repro_20260723`. The analytic RouteRL,
  persistence, and legacy metrics matched exactly. Neural-residual metrics
  differed by less than \(4\times10^{-11}\) in absolute value, consistent with
  floating-point reduction order and far below the 5% stochastic tolerance.
- Dense and sparse route-to-edge projection equivalence is covered by a unit
  test.

## Validation Report

- **Source**: `hdv_od_route_predictor_offline_20260723`
- **Overall Confidence**: CAUTION

The primary one-step results are measured on untouched chronological test
blocks and use paired time-series uncertainty. Confidence remains `CAUTION`
because each network contributes one evolving online run, the RouteRL anchor
is a specified behavioral update rather than an independently learned human
model, and completed-trip travel times can censor unfinished routes.

### Warnings

| Type | Detail | Affected |
|---|---|---|
| Single-run uncertainty | Bootstrap intervals describe episode dependence within one run, not training-seed or initial-state variation | All network comparisons |
| Aggregate inference | OD/edge distributions do not identify individual driver adaptation | Human-adaptation claims |
| Completion censoring | Only completed trips contribute realized route travel times | Route cost feedback |
| Stable-class dominance | High route top-1 accuracy partly reflects persistent dominant routes | Top-1 metric |
| Rule-based anchor | A successful RouteRL replay would validate the specified recurrence, not prove that a neural model learned human behavior | Mechanism interpretation |

### Fallacy Scan

- **Coverage**: 11/11 statistical fallacy types checked

| Fallacy | Severity | Assessment |
|---|---|---|
| Simpson's paradox | NOTE | Networks are reported separately; no pooled direction is used to hide the cross-network reversal versus persistence |
| Ecological fallacy | CAUTION | Aggregate OD/edge response cannot be interpreted as adaptation by individual drivers |
| Berkson's paradox | CAUTION | Completed-trip feedback is a selected subset when episodes contain unfinished vehicles |
| Collider bias | NOTE | No covariate-adjusted causal regression is used; the diagnostic does not condition on a post-treatment collider |
| Base-rate neglect | CAUTION | Route top-1 is interpreted alongside TV because the dominant-route base rate is high |
| Regression to the mean | NOTE | Splits are chronological and not selected from extreme error episodes |
| Survivorship bias | CAUTION | Unfinished route travel times are absent from the feedback signal |
| Look-elsewhere effect | NOTE | Edge TV versus persistence is predeclared as primary; other metrics are diagnostic and not separate discovery claims |
| Garden of forking paths | CAUTION | Alpha and residual scale are selected on validation; the final test block is untouched, but the grid remains a researcher choice |
| Correlation is not causation | CAUTION | One-step predictive agreement is not labeled a causal estimate of human adaptation |
| Reverse causality | CAUTION | Previous travel time temporally precedes the next route state, but shared congestion dynamics can still confound behavioral interpretation |
