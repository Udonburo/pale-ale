# Gate8A Execution And Scale-Up

Status: Tracked execution snapshot
Role: Tracked standing snapshot / execution-stage handoff
Date: 2026-03-19

## 0. Scope

This file records the first two Gate8 execution-stage reads:

- `gate8a_candidate_execution`: 16-row smoke
- `gate8b_128r_candidate_execution`: 128-row scale-up

The fixed comparison set remained:

- `F` / `score_F_gram_loop_v1`
- `gate6f` / `sigma_gap_tailkeep_weighted_gram_loop_v2`
- `gate6h` / `sigma_sqrtgap_tailkeep_object_v2`
- `gate7c` / `progression_anisotropic_closure_v3`

No new candidate was introduced.
No aggregation was introduced.

## 1. What Was Added

Gate8 now has an execution path, not only a benchmark scaffold.

New tooling:

- `tools/run_gate8_candidate_batch.py`
- `tools/evaluate_gate8_standing.py`
- `tools/run_gate8_scaleup.py`

The execution path now covers:

1. teacher-forced extraction on Gate8 answer targets
2. defect-span label materialization
3. Gate6 native local-span build
4. fixed-candidate execution
5. Gate8 conflict-cell and quietness evaluation

## 2. Smoke Read (`gate8a`)

On the 16-row smoke:

- `gate7c` became the strongest conflict candidate
- it exceeded `F` on both conflict cells
- quietness did not collapse

This was the first run where `gate7c` looked less like a merely mixed dynamic line and more like a candidate whose standing improves in retrieval-conflict geometry.

## 3. Scale-Up Read (`gate8b_128r`)

The 128-row run is the first real check against smoke fluke.

Headline read:

- `gate7c` still leads `F` on both conflict cells by `global_auprc`
- the margin is narrower than in smoke
- `F` still has slightly better `mean_delta_p90`
- `gate7c` has better `mean_top10_inflation`
- `gate6f` and `gate6h` remain clearly behind on this benchmark

So the correct sentence is:

- `gate7c` revival persists under scale-up, but as a narrowed standing-improvement signal, not yet a decisive universal reversal

## 4. Current Numerical Read

### 4.1 `gate8a` smoke

Direct contradiction:

- `gate7c global_auprc = 0.389995`
- `F global_auprc = 0.285101`

Distributed incompatibility:

- `gate7c global_auprc = 0.200154`
- `F global_auprc = 0.166861`

Quietness:

- `F mean_delta_p90 = -0.043442`
- `gate7c mean_delta_p90 = -0.009812`
- `F mean_top10_inflation = 2.250000`
- `gate7c mean_top10_inflation = 2.250000`

### 4.2 `gate8b_128r` scale-up

Direct contradiction:

- `gate7c global_auprc = 0.318939`
- `F global_auprc = 0.306034`

Distributed incompatibility:

- `gate7c global_auprc = 0.178790`
- `F global_auprc = 0.173201`

Quietness:

- `F mean_delta_p90 = -0.013795`
- `gate7c mean_delta_p90 = -0.005628`
- `F mean_top10_inflation = 2.593750`
- `gate7c mean_top10_inflation = 2.343750`

## 5. Caveats

These caveats remain active and should be stated explicitly in any external readout.

### 5.1 Smoke is not final standing

The 16-row run was evidence of reversal pressure, not proof of stable replacement.

### 5.2 Label granularity is not identical

- `gate7c` is evaluated on `label_transition`
- `F`, `gate6f`, and `gate6h` are evaluated on `label_token`

This is regime-consistent with Gate7, but it is not same-granularity comparison.

### 5.3 Quietness winner is still not settled

At 128 rows:

- `gate7c` does not collapse on quietness
- but it does not cleanly dominate `F`
- the `reachability` bucket is still the roughest quietness surface

So the right claim is:

- quietness survives
- quietness leadership does not yet transfer

## 6. Decision

The current decision should remain disciplined:

- do not add new candidates
- do not reopen the evaluator
- do not introduce aggregation rescue

The next responsible move is:

- scale Gate8 further under the same fixed set
- test whether `gate7c` keeps conflict-cell standing advantage
- test whether quietness remains non-collapsed at larger `n`

## 7. Working Sentence

The best short sentence after the first scale-up is:

- `Gate8 no longer reads as a smoke-only revival for gate7c; it now reads as a persistent but still caveated dynamic standing gain under retrieval-conflict geometry.`
