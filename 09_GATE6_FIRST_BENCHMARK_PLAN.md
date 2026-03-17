# Gate6 First Benchmark Plan

Status: Draft
Role: Tracked RFC / first benchmark plan for Gate6-A
Date: 2026-03-17

Naming note:
`Gate6-A` is a working label for the observation-redesign workstream only.
This document does not freeze later post-Gate5 workstreams into numbered gates.

## 0. Benchmark Philosophy

Gate6-A is not allowed to win by moving multiple things at once.

The first benchmark must isolate one question:

Does replacing FWHT-8D with native local span improve the same triad-loop experiment?

Therefore:

- same transport motif as Gate5
- same evaluation families as much as possible
- same CFA and Seam structure
- changed variable = observation boundary

## 1. Stage 0: Object Sanity Checks

Before any quality claims, verify object construction.

### 0.1 Reconstruction sanity

Using `B_t` and `C_t`, reconstruct local observables and verify low error:

- `||B_t @ C_t[:, 0] - v||`
- `||B_t @ C_t[:, 1] - p||`
- `||B_t @ C_t[:, 2] - m||`

Expected:

- near machine precision within the retained local rank

### 0.2 Rank profile

Report:

- fraction of rank-3
- fraction of rank-2
- fraction of rank-1

This matters because seam-heavy steps may collapse rank.

### 0.3 Gauge stability

Re-run with identical inputs.
Expect deterministic:

- `rank_local`
- `singular_values`
- `compat_local8`
- basis after sign-fix

### 0.4 Degeneracy visibility

No silent clamping.
If near-degenerate, it must appear in flags and rank profile.

## 2. Stage 1: Gate5 Motif Re-run Under New Boundary

Run the same triad loop motif as Gate5, but on `compat_local8`.

Compute:

- `rotor_loop_chordal_v1`
- any retained auxiliary loop residual
- the same downstream aggregation used in Gate5 where possible

Main comparison:

- Gate5 FWHT-8D triad loop
- Gate6-A native-span local8 triad loop

Primary question:

- does seam quietness improve?
- does CFA localization remain acceptable?

This is the only question Gate6-A v1 needs to answer.

## 3. Benchmark Sets

### 3.1 CFA

Purpose:

- defect sensitivity and localization

Use:

- the same CFA family used in Gate5

Key outputs:

- token-level AUPRC
- Hit@10
- first-hit distance
- representative case plots
- delta vs best baseline
- delta vs Gate5 triad loop

### 3.2 Seam Challenge

Purpose:

- seam false spike suppression

Use:

- clean consistent
- seam-perturbed consistent

Key outputs:

- mean delta max
- normalized IQR of delta max
- top-k inflation
- seam-span overlap
- representative seam cases

### 3.3 Promotion rule

Gate6-A does not promote because it is geometrically elegant.
For this RFC revision, it is promotion-eligible only under matched CFA and Seam slices with:

- identical sample ids
- identical model, layer, tokenizer, and baseline fields
- the same fixed triad-loop motif and downstream aggregation path

Promotion requires all of the following for the fixed Gate6-A triad-loop residual against the Gate5 FWHT triad-loop baseline on the same matched slice:

- CFA `global_auprc` drop is at most `0.010` absolute
- CFA `mean_sample_auprc` drop is at most `0.020` absolute
- Seam `mean_delta_max` is no worse than `0.005` absolute above baseline
- Seam `mean_iqr_normalized_delta_max` is no worse than `0.020` absolute above baseline
- at least one of `mean_delta_max` or `mean_iqr_normalized_delta_max` strictly improves

Canonical `F` remains a binding guardrail during the same comparison.
Promotion is invalid unless canonical `F` is emitted on the same matched slices and also satisfies all of the following against the Gate5 FWHT baseline:

- CFA `global_auprc_F` drop is at most `0.010` absolute
- CFA `mean_sample_auprc_F` drop is at most `0.020` absolute
- Seam `mean_delta_max_F` is no worse than `0.010` absolute above baseline
- Seam `mean_iqr_normalized_delta_max_F` is no worse than `0.050` absolute above baseline

These tolerances are binding for this RFC revision.
Changing them requires an explicit update to this document or a replacement standing contract.

## 4. Explicit Comparisons

### Allowed direct comparisons

- Gate5 triad-loop residual vs Gate6-A triad-loop residual
- Gate5 or FWHT-based `F` vs Gate6-A loop residual on CFA and Seam
- baselines vs Gate6-A

### Disallowed casual comparisons

- new boundary plus new motif vs old boundary plus old motif
- projector-level claims without transport comparison
- more geometric therefore better

## 5. Decision Table

### Outcome A: Win

- all section 3.3 thresholds satisfied

Action:

- Gate6-A becomes the new mainline boundary candidate
- proceed to the downstream transport motif comparison workstream

### Outcome B: Tradeoff

- Seam improves but CFA degrades
- or CFA improves but Seam worsens

Action:

- inspect the object family
- decide whether the tradeoff is structurally meaningful
- do not auto-promote

### Outcome C: Null

- no clear gain on Seam or CFA

Action:

- Gate6-A remains informative but not promoted
- revisit observable construction, not scalar post-processing

### Outcome D: Failure

- severe instability, gauge jitter, or collapse of localization

Action:

- reject the Gate6-A construction
- revise native object design

## 6. What Comes After Gate6-A Only If It Earns It

Only after Gate6-A earns promotion do we move to:

### Transport motif comparison

- transport motifs beyond the triad loop
- progression transport
- context closure
- cross-step motifs

### Field aggregation

- local defect field aggregation

### Benchmark expansion

- semi-closed retrieval conflict benchmark

The rule is simple:

- object first
- motif second
- field third
- broader benchmark fourth

Never reverse this order.

## 7. One-Sentence Test

Gate6-A passes only if this becomes true:

The same transport motif becomes more seam-stable under the native local boundary without losing its ability to localize contradiction-like defect.
