# Critic prediction repair context

User asks to determine how to change critic prediction logic. This turn is design and read-only validation; do not alter the live training job.

Evidence: frozen online checkpoint at `outputs/ingo_cohort_reward_audit_20261011/checkpoint.pt` is episode 345. Runtime source is `outputs/ingolstadt_full_retrain_20261009/source_paper_reward_20261010`. Eight observed-state checks give V about -56, six below physical finite-horizon limits. At t=2400, deadline=3810, feasible lower bounds about -23.4. Snapshot replay has 1008 eligible states; 64 (6.35%) can reach deadline within 240-second imagination. Current gamma=1, lambda=.95. Serial and parallel targets/terminal masks/detach/pooling agree; actual runtime uses eight workers. No existing online critic was inherited from the pretraining handoff.

Rewards remain fixed-cohort, no new births, no terminal penalty, disagreement weight zero. Actor entropy and prior changes are out of scope. Q baseline and macro V share physical mean-minutes return. Physical reward alone over a 240s rollout has magnitude at most four. Event training targets average around -54 in episodes 326-345; bootstrap dominates this scale. Bound validity requires no risk reward and no new births.

Touchpoints: `models/policy.py:CentralizedLatentValueCritic,lambda_return`; `models/event_critic.py:EventActionCritic,build_event_q_inputs`; `learning/training.py:_imagine_from_seed,_train_imagination_pass`; `learning/parallel_imagination.py:collect_parallel_imagination,_finish_pass`; `learning/replay.py:sample_active_seeds`; `workflow/checkpoint.py`, `workflow/config.py`. Existing audit artifacts in `outputs/ingo_cohort_reward_audit_20261011`.

Open issues: direct contribution of saturated features versus weak boundary grounding is not established; same-policy action rankings are not measured; oracle bounds do not establish useful credit assignment. Need fair alternatives, explicit value/target units, endpoint gradients and checkpoint incompatibility handling.
