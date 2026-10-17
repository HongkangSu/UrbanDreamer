# Condition-only B soft-filter tau sweep

Validation-only inference sensitivity, with FM weights and prior regression head fixed. All candidates use identical rank-16 graph noise, Z as source center and reference, and only change the background condition. This is not independent per-tau training.

32 states from 16 existing validation demand groups; K=8; Euler 100. Selection uses finite-K-corrected Energy Score. The old eight validation states reproduce the first four B draws within the preregistered 2e-4 tolerance.

| tau | fair ES ↓ | count fair CRPS ↓ | speed fair CRPS ↓ | count coverage90 | speed coverage90 |
|---|---:|---:|---:|---:|---:|
| 0.0 | 8.303703 | 0.480999 | 0.796784 | 0.407141 | 0.368945 |
| 0.01 | 8.303745 | 0.480999 | 0.796786 | 0.407038 | 0.369044 |
| 0.1 | 8.304029 | 0.480999 | 0.796795 | 0.407485 | 0.369943 |
| 1.0 | 8.304653 | 0.480973 | 0.796778 | 0.407864 | 0.369006 |
| 10.0 | 8.304382 | 0.480926 | 0.796616 | 0.406356 | 0.368963 |
| 100.0 | 8.304290 | 0.480912 | 0.796502 | 0.405969 | 0.369666 |

Lowest observed validation fair ES: tau=0.0; difference from tau=.1: -0.000325, paired demand-group bootstrap 95% CI [-0.000519, -0.000132].

Selection uncertainty is optimistic after searching six candidates. This sweep supports a deployment sensitivity check, not a universal optimum or downstream policy benefit.

Independent bank re-scoring: 1344 metrics passed; max error 5.33e-15.

Recommendation: keep tau=.1 as the mainline default. The maximum fair ES range across all six tau values is approximately 0.0114%; the apparent tau=0 improvement over .1 is only 0.0039%. Speed and count CRPS instead favor tau=100, with similarly small effects. No practically material jointly supported optimum was found.
