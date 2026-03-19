# Gate8A Execution And Scale-Up

Status: Tracked execution snapshot
Role: Tracked standing snapshot / execution-stage handoff
Date: 2026-03-19

## 0. Scope

This file records the corrected Gate8 execution read after the quietness-court repair
and the mixed-granularity court threading update.

The active evidence package is now:

- `gate8g_128r_granularity_candidate_execution`: 128-row same-world quietness court with explicit mixed-granularity artifacts
- `gate8h_200r_granularity_candidate_execution`: 200-row same-world quietness court with explicit mixed-granularity artifacts

The fixed comparison set remained:

- `F` / `score_F_gram_loop_v1`
- `gate6f` / `sigma_gap_tailkeep_weighted_gram_loop_v2`
- `gate6h` / `sigma_sqrtgap_tailkeep_object_v2`
- `gate7c` / `progression_anisotropic_closure_v3`

No new candidate was introduced.
No aggregation was introduced.

## 1. Court Correction

The earlier Gate8 execution path had a real quietness flaw.

Its quietness pairs matched:

- `clean_support`
- `surface_noisy_clean`

only by `world_type` and occurrence order.

That was not a valid same-world negative control.

So the old quietness sentence:

- `quietness did not collapse`

is withdrawn as a tracked claim.

The corrected court now requires:

- shared `world_id`
- distinct `rendering_id`
- pairing rule `shared_world_id_v1`

This fixes the quietness readout to mean:

- same underlying world
- different surface realization

rather than:

- merely same coarse world type
- different world instance

The fixed standing court also now carries explicit mixed-granularity metadata:

- per-candidate `candidate_id`
- per-candidate `role`
- per-candidate `label_key`
- per-candidate `label_granularity`
- run-level status `mixed_candidate_label_granularity_v1`

That does not remove the caveat.

It makes the caveat impossible to silently drop from the artifacts.

## 2. What Was Added

Gate8 now has:

- fixed-set execution
- corrected same-world quietness controls
- scale-up evidence through 200 rows
- explicit mixed-granularity court threading across constitution, materialization, execution, and per-candidate evaluation

New tooling and contract tightening live in:

- `tools/run_gate8_candidate_batch.py`
- `tools/evaluate_gate8_standing.py`
- `tools/run_gate8_scaleup.py`
- `15_GATE8_LABEL_AND_PROVENANCE_RULES.md`

## 3. Corrected Read

Under the corrected same-world quietness court:

- `gate7c` still leads `F` on both conflict cells
- this remains true at both 128 and 200 rows
- `gate6f` and `gate6h` remain clearly behind
- the numerical standing read is unchanged by the granularity-threading rerun

But quietness is still not fully won:

- `F` remains better on `mean_delta_p90`
- `gate7c` remains better on `mean_top10_inflation`

So the correct sentence is:

- `gate7c` conflict-side revival persists under the corrected court, while quietness remains cleaner than before but still unresolved

## 4. Current Numerical Read

### 4.1 `gate8g_128r_granularity`

Direct contradiction:

- `gate7c global_auprc = 0.318939`
- `F global_auprc = 0.306034`

Distributed incompatibility:

- `gate7c global_auprc = 0.179488`
- `F global_auprc = 0.169906`

Quietness:

- `F mean_delta_p90 = -0.020208`
- `gate7c mean_delta_p90 = -0.007069`
- `F mean_top10_inflation = 2.031250`
- `gate7c mean_top10_inflation = 1.906250`

### 4.2 `gate8h_200r_granularity`

Direct contradiction:

- `gate7c global_auprc = 0.316612`
- `F global_auprc = 0.300950`

Distributed incompatibility:

- `gate7c global_auprc = 0.167865`
- `F global_auprc = 0.160062`

Quietness:

- `F mean_delta_p90 = -0.021322`
- `gate7c mean_delta_p90 = -0.006514`
- `F mean_top10_inflation = 2.060000`
- `gate7c mean_top10_inflation = 1.980000`

## 5. Caveats

These caveats remain active and should be stated explicitly in any external readout.

### 5.1 Pre-qfix quietness is historical only

The old 16-row and pre-qfix 128-row quietness claims should not be used as current evidence.

Their conflict-side direction is still historically interesting.
Their quietness court is not valid enough for tracked outward claims.

### 5.2 Label granularity is not identical

- `gate7c` is evaluated on `label_transition`
- `F`, `gate6f`, and `gate6h` are evaluated on `label_token`

This is regime-consistent with Gate7, but it is not same-granularity comparison.

This caveat is now carried explicitly in:

- constitution artifacts
- materialized benchmark manifests
- execution manifests
- per-candidate evaluation manifests and reports

### 5.3 Quietness winner is still not settled

Under the corrected court:

- `gate7c` is no longer being protected by the old pairing confound
- but it still does not cleanly dominate `F`

So the right claim is:

- quietness is better adjudicated
- quietness leadership still does not transfer

## 6. Decision

The current decision should remain disciplined:

- do not add new candidates
- do not reopen the evaluator
- do not introduce aggregation rescue

What is now earned:

- the `gate7c` conflict-side revival does not depend on the old quietness bug

What is not yet earned:

- a full quietness victory claim
- a settled dynamic mainline replacement claim

## 7. Working Sentence

The best short sentence after the quietness-court correction is:

- `Under the corrected same-world quietness court and explicit mixed-granularity standing metadata, gate7c retains a persistent conflict-side standing gain through 200 rows, while quietness remains non-collapsed in some respects but still unresolved overall.`
