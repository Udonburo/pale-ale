# Gate7 Progression Leak Smoke

Status: Draft
Role: Tracked smoke result / implementation-facing
Date: 2026-03-18

## 0. Scope

This document records the first dynamic Gate7 smoke after the Gate6 closeout.

The question is intentionally narrow:

- does a projector-native step-to-step progression motif carry useful signal before any field aggregation?

The tested primary metric is:

- `progression_leak_v1 = 1 - ||P_t v_{t+1}||^2 / ||v_{t+1}||^2`

Where:

- `P_t` is the current local projector from the Gate6 native object
- `v_{t+1}` is the next-step anchor direction reconstructed from the next local object

This is the minimal object-native dynamic motif.
It is not yet a field.
It is not yet persistent topology.
It is not yet a benchmark policy layer.

Coverage note:

- the terminal row of each sample is structurally non-evaluable and is tracked as `final_step_no_successor`
- this structural case is recorded separately from true invalid or missing transitions

## 1. Artifact Surface

Primary artifacts:

- [`runs/gate7a_cfa_full/gate7a_aggregate_summary.md`](runs/gate7a_cfa_full/gate7a_aggregate_summary.md)
- [`runs/gate7a_seam_full/gate7a_aggregate_summary.md`](runs/gate7a_seam_full/gate7a_aggregate_summary.md)
- [`runs/gate7a_seam_pairs_full/gate7a_seam_report.md`](runs/gate7a_seam_pairs_full/gate7a_seam_report.md)

Builder and regression:

- [`tools/run_gate7_progression_leak_consumer.py`](tools/run_gate7_progression_leak_consumer.py)
- [`tools/test_run_gate7_progression_leak_consumer.py`](tools/test_run_gate7_progression_leak_consumer.py)

## 2. CFA Read

Headline comparison on matched CFA full artifacts:

- `global_auprc_score_F_gram_loop_v1 = 0.160699`
- `global_auprc_progression_leak_v1 = 0.154030`
- `mean_sample_auprc_score_F_gram_loop_v1 = 0.378766`
- `mean_sample_auprc_progression_leak_v1 = 0.379663`
- `mean_hit@10_score_F_gram_loop_v1 = 2.760000`
- `mean_hit@10_progression_leak_v1 = 2.780000`

Read:

- global ranking is slightly worse than the current `F` guardrail
- sample-mean AUPRC is slightly better
- hit@10 is slightly better

This is not a clean CFA win.
It is also not a collapse.

## 3. Seam Read

Paired quietness against the same `F` guardrail:

- `mean_delta_max_score_F_gram_loop_v1 = 0.169693`
- `mean_delta_max_progression_leak_v1 = -0.026743`
- `mean_delta_p90_score_F_gram_loop_v1 = 0.049965`
- `mean_delta_p90_progression_leak_v1 = 0.178997`
- `mean_iqr_normalized_delta_max_score_F_gram_loop_v1 = 1.414533`
- `mean_iqr_normalized_delta_max_progression_leak_v1 = -0.419330`
- `mean_top10_inflation_score_F_gram_loop_v1_vs_clean_p90 = 2.390625`
- `mean_top10_inflation_progression_leak_v1_vs_clean_p90 = 2.437500`

Read:

- `mean_delta_max` is clearly quieter than the guardrail
- `mean_iqr_normalized_delta_max` is also clearly quieter
- `mean_delta_p90` is worse
- `mean_top10_inflation` is slightly worse

So the motif is not dead.
It is mixed in exactly the way a first dynamic projector motif is allowed to be mixed.

## 4. Decision

Current standing:

- status: `mixed keep`
- promotion: no
- field aggregation unlock: not yet

Interpretation:

- projector progression leakage is a real object-native dynamic signal
- it is not yet strong enough to justify skipping directly to field construction
- the next narrow step should tune or contrast dynamic projector motifs before adding persistence machinery

## 5. What This Changes

Gate6 fixed the static object layer.

Gate7a now establishes that:

- the first dynamic object-native motif is viable
- `P_t -> V_{t+1}` leakage can be evaluated on the same CFA and Seam surfaces
- progression dynamics should be the next mainline focus, not benchmark policy or retrieval conflict

## 6. Immediate Next Step

The next Gate7 unit should remain narrow:

- keep the same datasets
- keep the same CFA aggregate and Seam pair evaluator
- change only the dynamic transport law

Recommended focus:

- another projector-native progression motif that tries to preserve the quietness gains in `mean_delta_max` and `iqr_normalized_delta_max`
- while reducing the current `p90` and `top10_inflation` regression
