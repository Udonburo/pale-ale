# Gate8A Execution And Scale-Up

Status: Tracked execution snapshot
Role: Tracked standing snapshot / execution-stage handoff
Date: 2026-03-20

## 0. Scope

This file records the corrected Gate8 execution read after the quietness-court repair,
the mixed-granularity court threading update, and the first diagnostic-only
rotation/leakage bridge rerun, and the first support/closure bridge burn on the fixed Gate8 benchmarks.

The active evidence package is now:

- `gate8i_128r_bridge_candidate_execution`: 128-row fixed-benchmark rerun with explicit mixed-granularity artifacts and diagnostic-only bridge outputs
- `gate8j_200r_bridge_candidate_execution`: 200-row fixed-benchmark rerun with explicit mixed-granularity artifacts and diagnostic-only bridge outputs
- `gate8k_128r_support_closure_candidate_execution`: 128-row fixed-benchmark rerun with added `support_anchor_coverage` / `support_reanchor_cost` / `support_conditioned_closure`
- `gate8l_200r_support_closure_candidate_execution`: 200-row fixed-benchmark rerun with added `support_anchor_coverage` / `support_reanchor_cost` / `support_conditioned_closure`

These execution reruns reuse the already-fixed benchmarks:

- `gate8g_128r_granularity_benchmark`
- `gate8h_200r_granularity_benchmark`

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
- diagnostic-only bridge outputs for `rotation_only`, `leakage_only`, and `closure_defect`
- diagnostic-only bridge outputs for `support_anchor_coverage`, `support_reanchor_cost`, and `support_conditioned_closure`
- artifact-level bridge failure read in the emitted diagnostic report itself

New tooling and contract tightening live in:

- `tools/run_gate8_candidate_batch.py`
- `tools/evaluate_gate8_standing.py`
- `tools/run_gate8_scaleup.py`
- `15_GATE8_LABEL_AND_PROVENANCE_RULES.md`
- `18_GATE8_ROTATION_LEAKAGE_BRIDGE.md`

## 3. Corrected Read

Under the corrected same-world quietness court:

- `gate7c` still leads `F` on both conflict cells
- this remains true at both 128 and 200 rows
- `gate6f` and `gate6h` remain clearly behind
- the numerical standing read is unchanged by the granularity-threading rerun

But quietness is still not fully won:

- `F` remains better on `mean_delta_p90`
- `gate7c` remains better on `mean_top10_inflation`

The bridge diagnostics are now baked on real execution artifacts too.

But the first bridge read is not clean enough to license a story win:

- `rotation_only` stays high across all four cells
- `leakage_only` is not uniquely quiet on clean/noisy cells and is in fact lowest on `direct_contradiction`
- `closure_defect` shows only weak tail elevation on `distributed_incompatibility` and is not cleanly separated from `surface_noisy_clean`

The second bridge burn is also not a full explanatory win:

- `distributed_incompatibility` does become the lowest-coverage / highest-reanchor / highest-closure-tail cell
- `clean_support` and `surface_noisy_clean` remain close to each other, which is directionally acceptable
- but `direct_contradiction` does not rise on `support_conditioned_closure`; it is actually lower than the clean/noisy cells on mean closure

So the correct sentence is now:

- `gate7c` conflict-side revival persists under the corrected court, while quietness remains unresolved, bridge v1 stays a clean negative, and bridge v2 only partially cuts the taxonomy by making distributed incompatibility more legible without carrying direct contradiction with it.`

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

Bridge diagnostics:

- `rotation_only` mean is high across all cells: `surface_noisy_clean 0.597033`, `clean_support 0.578749`, `distributed_incompatibility 0.571634`, `direct_contradiction 0.545732`
- `leakage_only` mean is lowest on `direct_contradiction 0.253358`; the other three cells cluster around `0.299-0.309`
- `closure_defect` mean stays broad across all cells (`0.499022-0.519938`); `distributed_incompatibility` has the highest `p90 = 0.608549`, but `surface_noisy_clean` is close at `0.606212`

Support/closure bridge:

- `distributed_incompatibility` is the lowest-coverage cell at `0.655203` and the highest-reanchor cell at `0.588949`
- `distributed_incompatibility` also has the highest `support_conditioned_closure p90 = 0.927918`
- but `direct_contradiction mean_support_conditioned_closure = 0.764955` remains below `clean_support = 0.798421` and `surface_noisy_clean = 0.798094`

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

Bridge diagnostics:

- `rotation_only` mean is high across all cells: `surface_noisy_clean 0.597114`, `clean_support 0.578824`, `distributed_incompatibility 0.571218`, `direct_contradiction 0.545658`
- `leakage_only` mean is again lowest on `direct_contradiction 0.252066`; `clean_support`, `surface_noisy_clean`, and `distributed_incompatibility` sit at `0.299787-0.307476`
- `closure_defect` mean remains broad (`0.500191-0.519174`); `distributed_incompatibility` has the highest `p90 = 0.609673`, but `surface_noisy_clean` remains close at `0.607331`

Support/closure bridge:

- `distributed_incompatibility` is again the lowest-coverage cell at `0.653657` and the highest-reanchor cell at `0.591002`
- `distributed_incompatibility` also has the highest `support_conditioned_closure p90 = 0.924696`
- but `direct_contradiction mean_support_conditioned_closure = 0.763350` remains below `clean_support = 0.798916` and `surface_noisy_clean = 0.798365`

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

### 5.4 Bridge diagnostics are baked, not won

The bridge outputs now exist on real 128-row and 200-row execution artifacts.

That earns:

- a falsifiable bridge surface on the fixed court
- manifest-bound diagnostic outputs beside the standing artifacts
- diagnostic reports that explicitly state what bridge v1 failed to separate

That does not earn:

- a clean `rotation vs leakage vs closure_defect` separation claim
- a promote / replace conclusion for `gate7c`
- a doctrinal victory sentence about lawful jump versus unlawful escape

## 6. Decision

The current decision should remain disciplined:

- do not add new candidates
- do not reopen the evaluator
- do not introduce aggregation rescue
- do not promote bridge diagnostics into standing metrics

What is now earned:

- the `gate7c` conflict-side revival does not depend on the old quietness bug
- the bridge diagnostics are now real execution artifacts, not doctrine-only placeholders

What is not yet earned:

- a full quietness victory claim
- a settled dynamic mainline replacement claim
- a clean bridge-level explanation of Seam-tail burden
- a closure-centric contradiction read that survives both conflict cells

The next move, if any, should therefore remain narrow:

- do not rescue `bridge v1`
- do not promote `bridge v2`
- read `bridge v2` as partial / mixed, not as explanatory settlement
- if another bridge is tried later, it should start from the direct-vs-distributed split that `v2` failed to unify

## 7. Working Sentence

The best short sentence after the bridge rerun is:

- `Under the corrected same-world quietness court and explicit mixed-granularity standing metadata, gate7c retains a persistent conflict-side standing gain through 200 rows, while quietness remains unresolved, bridge v1 remains a clean negative, and bridge v2 yields only a partial read in which distributed incompatibility sharpens but direct contradiction does not survive as closure-first burden.`
