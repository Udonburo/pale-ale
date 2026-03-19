# Gate8 Support-Conditioned Closure Bridge

Status: spec-only draft
Role: diagnostic bridge v2 spec, not standing spec
Date: 2026-03-20

## 0. Why This Exists

`bridge v1` closed cleanly.

It taught two things at once:

- `gate7c` standing revival is still real on the fixed court
- `rotation/leakage/closure_defect v1` is not the explanatory cut that makes that revival scientifically legible

The next bridge should therefore narrow rather than expand.

This document chooses one bridge only.

## 1. Bridge Choice

The first-class bridge for v2 is:

- `closure-centric contradiction read`

Within that bridge, `support-conditioned re-anchoring` is not a competing story.

It is the computational precondition.

In other words:

- first align the answer-side state to an explicit support-conditioned anchor
- then ask whether contradiction still survives as failure to close under that re-anchored transport

This bridge does not reopen the old leakage-first story.

## 2. Court Discipline

This bridge inherits the fixed Gate8 court unchanged.

The following remain fixed:

- candidate set: `F`, `gate6f`, `gate6h`, `gate7c`
- standing evaluator
- conflict taxonomy
- quietness court

The following remain forbidden:

- adding a new ranking candidate
- modifying the standing table
- introducing rescue aggregation
- narrating promote / replace conclusions from bridge diagnostics alone

## 3. Working Hypothesis

The working hypothesis is:

- some Seam-tail burden comes from answer motion that only looks anomalous until the answer is re-anchored to the support it is actually trying to use
- after that re-anchoring, clean and surface-noisy samples should remain largely closable
- contradiction-like samples should still fail to close even after the support-conditioned anchor is granted

So the central question is not:

- did the state move a lot

The central question is:

- after support-conditioned re-anchoring, does the transport still fail to close

## 4. Objects

This bridge introduces one primary readout and two support diagnostics.

They are not ranking metrics.

### 4.1 `support_anchor_coverage`

Intended meaning:

- how much explicit support-conditioned anchor was actually available for the sample

Intended use:

- prevent post hoc bridge success stories on samples where the anchor was never really defined

### 4.2 `support_reanchor_cost`

Intended meaning:

- how much motion is required to align the answer-side local view to the chosen support-conditioned anchor

Intended use:

- distinguish benign support re-anchoring burden from contradiction-like non-closure

### 4.3 `support_conditioned_closure`

Intended meaning:

- residual non-closure that remains after support-conditioned re-anchoring is granted

Intended use:

- make contradiction read closure-first rather than leakage-first

This is the primary object of the bridge.

## 5. Placement Rules

These outputs must remain on a separate surface from standing.

Allowed placements:

- per-sample diagnostic CSV or JSON
- per-cell aggregate summary
- separate markdown bridge report

Forbidden placements:

- insertion into `candidate_summary.csv`
- insertion into the standing headline table
- any rescue score that mixes bridge outputs back into the fixed court

## 6. Cell Expectations

These expectations must be fixed before implementation.

### 6.1 `clean_support`

Expected:

- `support_anchor_coverage` should be high enough to make the bridge meaningful
- `support_reanchor_cost` may be non-trivial
- `support_conditioned_closure` should remain low

Interpretation:

- support may need to be re-entered
- but the path should still close once that support is granted

### 6.2 `surface_noisy_clean`

Expected:

- `support_anchor_coverage` should remain high
- `support_reanchor_cost` may rise relative to `clean_support`
- `support_conditioned_closure` should remain materially closer to `clean_support` than to contradiction cells

Interpretation:

- surface wobble may make re-anchoring harder
- it should not by itself create contradiction-like non-closure

### 6.3 `direct_contradiction`

Expected:

- `support_anchor_coverage` may still be adequate
- `support_reanchor_cost` alone should not explain the cell
- `support_conditioned_closure` should rise materially

Interpretation:

- once support is granted, contradiction should still survive as failure to close

### 6.4 `distributed_incompatibility`

Expected:

- `support_anchor_coverage` may be patchier than in direct contradiction
- `support_conditioned_closure` should still rise materially
- the closure burden may appear more broadly or persistently than in `direct_contradiction`

Interpretation:

- distributed conflict should remain a closure problem, not just a re-anchoring burden

## 7. Falsifiers

This bridge weakens or fails under any of the following:

- `clean_support` shows ordinary high `support_conditioned_closure`
- `surface_noisy_clean` remains close to contradiction cells on `support_conditioned_closure`
- `direct_contradiction` is explained mainly by `support_reanchor_cost` without clear closure elevation
- `distributed_incompatibility` does not show broader or stronger closure-style burden than the clean cells
- `support_anchor_coverage` is too weak, unstable, or benchmark-shaped to make the bridge legible

More sharply:

- if support conditioning does not quiet the clean/noisy side, the bridge has not earned its premise
- if contradiction does not survive support conditioning as closure failure, the closure-centric story weakens

## 8. Non-Promotion Rule

Even a successful v2 bridge does not immediately earn:

- candidate promotion
- replacement of `gate7c`
- a new standing metric
- a field-level story

At most, it would earn:

- the right to prototype one later diagnostic or candidate that is explicitly support-conditioned and closure-first

## 9. Failure Reading

If this bridge fails, the honest readings include:

- support-conditioned re-anchoring is still not the right explanatory cut
- contradiction on this boundary is not best read through this closure construction
- the current observation boundary may still be too compressed for explanatory bridge work

Failure here does not license:

- adding more diagnostics at once
- broadening the court
- narrative rescue

## 10. Exit Condition

This bridge phase is ready for implementation only if:

- the bridge stays single-threaded around `support_conditioned_closure`
- support-conditioned re-anchoring is treated as precondition, not as a second bridge
- the four cell expectations are frozen up front
- the falsifiers are written before code

Until then, this remains spec-only.
