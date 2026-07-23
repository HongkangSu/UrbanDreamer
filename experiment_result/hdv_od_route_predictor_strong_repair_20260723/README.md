# Persistence-anchored HDV OD-route response predictor

This directory contains the corrected, chronological offline comparison for
the persistence-anchored OD-route response model. All reported test metrics
come from a held-out final episode block. Training, early stopping, Gawron
calibration, gate scaling, and model-family selection use only the preceding
training and validation blocks.

## Corrected protocol

- HDV route labels use only valid `hdv_action` values from seen, successfully
  route-matched Human vehicles. They never fall back to
  `initial_route_action`.
- Travel-time replay uses Human `hdv_action` and AV `policy_action`, matching
  the online collector. Invalid actions and unseen vehicles do not contribute.
- Raw HDV route counts and travel-time sample counts are retained.
- Route persistence is the per-OD last observed empirical histogram, carried
  forward causally. An OD with no observation yet uses the segment-start
  latent route state; an empty OD is never replaced by a uniform pseudo-label.
- The learned model predicts an OD-local gate between this strong persistence
  anchor and the exact Gawron proposal.
- Population-weighted OD-route TV is the primary selection metric. Edge TV is
  a guard metric, computed by applying the projected route response to the
  exact previous edge context.
- A non-persistence candidate must clear the pre-registered validation route
  gain, route win-rate, and edge degradation guards. Otherwise the saved
  checkpoint has an exact zero gate.

The strict route reconstruction matches the archived edge target with mean TV
between `1.53e-7` and `1.86e-7` across the three networks.

## Held-out test results

TV values and confidence intervals below are multiplied by \(10^3\). Deltas
are candidate minus persistence, so negative is better.

| Network | Selected | Route persistence | Exact Gawron route TV | Learned gate route TV | Gate route delta [95% CI] | Edge persistence | Exact Gawron edge TV | Learned gate edge TV | Gate edge delta [95% CI] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gretz-Armainvilliers | Persistence | 0.338 | 2.721 | 0.338 | 0.000 [0.000, 0.000] | 4.171 | 3.983 | 4.171 | 0.000 [0.000, 0.000] |
| Ingolstadt | Persistence | 0.075 | 1.714 | 0.075 | 0.000 [0.000, 0.000] | 0.183 | 0.172 | 0.183 | 0.000 [0.000, 0.000] |
| Fontenay-en-Parisis | Learned gate | 1.222 | 2.499 | **0.692** | **-0.530 [-0.770, -0.330]** | 3.773 | 3.233 | **3.174** | **-0.600 [-0.882, -0.433]** |

The deployed legacy edge MLP has edge TV `10.505`, `2.223`, and `23.065`
(\(\times 10^3\)) on Gretz, Ingolstadt, and Fontenay, respectively.

## Interpretation

The stronger repair produces a genuine, validation-selected improvement on
Fontenay: route and edge TV both beat last-observed persistence with paired
moving-block bootstrap intervals entirely below zero. Gretz and Ingolstadt
contain no validation route change under the strong persistence definition,
so the learned gate correctly collapses to zero instead of fabricating a
response.

This is evidence for a useful learned response correction on Fontenay, not a
cross-network proof of human adaptation. The target is conditioned on
successful collector registration and route matching, and the evaluation is
prequential one-step prediction with observed previous-episode feedback, not
an open-loop multi-episode forecast. Multi-seed paired simulator interventions
remain necessary for a broad causal behavioral claim.
