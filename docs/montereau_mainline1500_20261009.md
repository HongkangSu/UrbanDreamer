# Montereau demand mainline, 2026-10-09

The accepted mainline uses `occupancy1500`: 1500 expected Poisson vehicles over
1800 seconds, with 283 positive-demand OD pairs. Actual episode populations vary.
Only `networks/montereau_fault_yonne/agents.csv` changes. Network geometry,
connectivity, OD index file, and edge registry remain unchanged. Incidents remain
disabled and shortest-path prior remains zero. Training deadline remains 3810 s.

Candidate and evaluation evidence: `outputs/demand_od_screen_20261009/`.
Deployment hashes, original inputs, and historical link pinning are recorded in
`outputs/montereau_mainline1500_20261009/deployment.json`.
The queue and current DTA evaluator now expect 1500 vehicles for new runs.
The input promotion initially left the old 2806-vehicle batch unchanged.
At 2026-10-09 17:52 UTC the user-authorized restart stopped the old C/M jobs
and started fresh learning runs in
`outputs/city_baselines_v5_1500_noprior_penalty_20261009`.
Coulommiers now uses v5/1811; Montereau uses occupancy1500 with its tested
original geometry. DTA and Random are excluded; all Ingo Random sampling was
also stopped. Other Ingo learning jobs retain their original PIDs and outputs.
The new queue owns scheduling; never resume the old stopped controllers.

On three held-out demand seeds, shortest path and five-iteration DTA both
completed all vehicles. Mean departure-to-arrival time was 155.058 s versus
112.486 s (27.456% improvement). Fresh stochastic AC completion was 19.12%
across three seeds: this remains a difficult cold-start training setting.

Do not regenerate this demand with `prepare_ingo_reference_candidates.py`:
that historical importer starts from the original 3000-vehicle distribution.
The accepted reproducible input is the candidate bundle at
`outputs/demand_od_screen_20261009/candidates/occupancy1500/`.

Reward semantics are unchanged: shared reward is negative observed active
vehicle minutes divided by total episode population. Nonarrival cost is -5000
per failed vehicle at the real deadline. AC's global critic excludes that cost;
its actor adds the individual cost after scaling shared advantages, then clips
to [-2.5, 2.5]. Episode selection score adds the total nonarrival cost to the
shared episode reward. Vehicles without decision rows have no direct actor
penalty sample, although they are counted in the selection score.
