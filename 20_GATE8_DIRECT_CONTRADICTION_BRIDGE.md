# Gate8 Direct Contradiction Bridge

Status: spec-only draft
Role: diagnostic bridge v3 spec, not standing spec
Date: 2026-03-20

## 0. Why This Exists

`bridge v2` was a partial gain, not a settlement.

It made `distributed_incompatibility` somewhat more legible.

It did not make `direct_contradiction` legible in the way the bridge claimed.

The next bridge should therefore narrow again.

This document does not try to rescue all of Gate8 at once.

It chooses one unresolved locus only:

- `direct_contradiction`

## 1. Narrowing Rule

This bridge is not judged on a blended `direct_contradiction` cell mean.

It must explicitly split:

- `direct_contradiction / consistent_answer`
- `direct_contradiction / conflict_following_wrong_answer`

If those two answer target types are mixed back together, the bridge becomes scientifically unreadable.

This bridge does not try to solve:

- `distributed_incompatibility`
- quietness court
- token/transition court reconciliation
- candidate promotion

## 2. Court Discipline

The fixed Gate8 court remains unchanged.

The following remain fixed:

- candidate set: `F`, `gate6f`, `gate6h`, `gate7c`
- standing evaluator
- mixed-granularity caveat
- conflict taxonomy

The following remain forbidden:

- adding a new ranking candidate
- modifying the standing table
- mixing bridge outputs back into fixed-court scoring
- using this bridge as a promote / replace argument by itself

## 3. Bridge Choice

The first-class bridge for this phase is:

- `dual_anchor_contradiction_gap`

This bridge uses two operational anchors:

- `support anchor`
- `conflict anchor`

But those are not rival headline metrics.

They are the minimum structure needed to define the single primary read.

## 4. Working Hypothesis

The working hypothesis is:

- in `direct_contradiction / consistent_answer`, the answer-side state should still remain more support-closable than conflict-closable
- in `direct_contradiction / conflict_following_wrong_answer`, the answer-side state should shift toward the conflict anchor in a way that is visible after the dual-anchor comparison

So the central question is not:

- did contradiction exist somewhere in the prompt

The central question is:

- which anchor the answer-side motion closes toward when both anchors are made explicit

## 5. Objects

This bridge has one primary readout and two support diagnostics.

They remain diagnostic-only.

### 5.1 `support_anchor`

Intended meaning:

- the anchor induced by the aligned support path in the `direct_contradiction` prompt

Intended use:

- define the lawful answer-side reference

### 5.2 `conflict_anchor`

Intended meaning:

- the anchor induced by the explicit contradictory memo in the same prompt

Intended use:

- define the rival contradiction-side reference

### 5.3 `dual_anchor_contradiction_gap`

Working definition:

- `dual_anchor_contradiction_gap := support_conditioned_closure - conflict_conditioned_closure`

Sign convention:

- positive means the state closes more readily toward the conflict anchor than the support anchor
- negative means the state closes more readily toward the support anchor than the conflict anchor
- near-zero means the bridge has little directional preference

This is the only primary object of the bridge.

### 5.4 Support Diagnostics

Allowed support diagnostics:

- `support_anchor_coverage`
- `conflict_anchor_coverage`

Intended use:

- prevent fake bridge wins caused by one anchor being undefined, weak, or coverage-collapsed

## 6. Placement Rules

These outputs must remain on a separate surface from standing.

Allowed placements:

- per-sample diagnostic CSV or JSON
- per-answer-target-type aggregate summary inside `direct_contradiction`
- separate markdown bridge report

Forbidden placements:

- insertion into `candidate_summary.csv`
- insertion into the standing headline table
- rescue aggregation back into the fixed court

## 7. Expectations

These expectations must be frozen before implementation.

### 7.1 `direct_contradiction / consistent_answer`

Expected:

- `support_anchor_coverage` should be adequate enough to make the comparison legible
- `conflict_anchor_coverage` may be non-zero because the contradiction is explicit in the prompt
- `dual_anchor_contradiction_gap` should remain support-favoring or at least non-conflict-favoring

Interpretation:

- the prompt contains contradiction
- but the answer does not surrender to it

### 7.2 `direct_contradiction / conflict_following_wrong_answer`

Expected:

- both anchors should be legible enough for comparison
- `dual_anchor_contradiction_gap` should shift materially toward the conflict side

Interpretation:

- the wrong answer should not merely be noisy
- it should look more conflict-closable than support-closable

## 8. Falsifiers

This bridge weakens or fails under any of the following:

- `dual_anchor_contradiction_gap` does not directionally separate `consistent_answer` from `conflict_following_wrong_answer`
- `consistent_answer` shows ordinary positive gap similar to the wrong-answer split
- `conflict_following_wrong_answer` does not show a clear conflict-side shift
- the apparent split is explained mainly by one anchor having weak or collapsed coverage
- the bridge only works after threshold theater, benchmark-shaped filtering, or post hoc grouping

More sharply:

- if `consistent_answer` and `conflict_following_wrong_answer` do not separate within `direct_contradiction`, this bridge has failed its reason for existing
- if the wrong-answer split does not prefer the conflict anchor, the dual-anchor story weakens directly

## 9. What Stays Out

This bridge is not allowed to quietly expand into:

- `distributed_incompatibility` rescue
- quietness rescue
- new standing metrics
- field or aggregation language
- token/transition court unification

Those are different responsibilities.

They should not be smuggled into this bridge.

## 10. Non-Promotion Rule

Even a successful direct-contradiction bridge does not immediately earn:

- candidate promotion
- replacement of `gate7c`
- a new standing metric
- a global explanation of all Gate8 conflict cells

At most, it would earn:

- the right to say that `direct_contradiction` has a legible dual-anchor diagnostic cut on the current boundary

## 11. Failure Reading

If this bridge fails, the honest readings include:

- direct contradiction on this boundary is not best read as dual-anchor closure preference
- the present observation boundary still compresses away the needed anchor structure
- the contradiction signal may live in a different motif than this bridge assumes

Failure here does not license:

- broader story inflation
- more bridge objects at once
- candidate rescue

## 12. Exit Condition

This bridge is ready for implementation only if:

- `direct_contradiction` is split by `answer_target_type` from the start
- `dual_anchor_contradiction_gap` remains the single primary object
- anchor coverage is treated as support hygiene, not as a second headline story
- falsifiers are written before code
- the non-promotion rule remains explicit

Until then, this remains spec-only.
