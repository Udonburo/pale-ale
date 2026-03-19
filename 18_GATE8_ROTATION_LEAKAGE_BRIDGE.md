# Gate8 Rotation Leakage Bridge

Status: draft skeleton
Role: diagnostic bridge spec, not standing spec
Date: 2026-03-19

## 0. Purpose

This document defines the first Gate8 bridge experiment for the hypothesis that:

- `gate7c` conflict-side signal is real
- its remaining Seam-tail weakness may come from mixing lawful contextual rotation with unlawful leakage or closure failure

The job of this document is not to create a new ranking candidate.

Its job is to define a falsifiable diagnostic bridge on the fixed Gate8 court.

## 1. Court Discipline

This bridge inherits the Gate8 fixed court without reopening it.

The following remain fixed:

- candidate set: `F`, `gate6f`, `gate6h`, `gate7c`
- standing evaluator
- conflict taxonomy
- quietness court

The following are explicitly forbidden in this bridge phase:

- adding a new ranking candidate
- replacing `gate7c`
- promoting a diagnostic into the standing table
- aggregating diagnostics into a rescue score
- narrating promote / replace conclusions directly from bridge diagnostics

## 2. Bridge Status

This is a diagnostic bridge only.

That means:

- outputs live beside the standing court, not inside it
- diagnostics are explanatory probes, not new winners
- failure of the bridge is allowed and informative

## 3. Working Hypothesis

The working hypothesis is:

- some high Seam-tail events are lawful contextual updates
- those updates may look large in latent motion without being genuine defect
- current dynamic readouts may not cleanly separate that lawful motion from actual breakdown

In short:

- lawful jump = rotation-like update
- unlawful jump = leakage or closure defect

This is a hypothesis to break, not a truth to defend.

## 4. Diagnostic Readouts

The bridge introduces exactly three diagnostic readouts.

They are not ranking metrics.

### 4.1 `rotation_only`

Intended meaning:

- motion that is large but remains explainable as lawful local frame change
- contextual redirection, support re-anchoring, or benign transport update

Intended use:

- explain high movement without immediately calling it defect

### 4.2 `leakage_only`

Intended meaning:

- motion that escapes the expected local frame or span
- behavior suggestive of orthogonal drift, unsupported departure, or bad continuation

Intended use:

- distinguish abnormal escape from lawful contextual change

### 4.3 `closure_defect`

Intended meaning:

- residual failure that remains after lawful motion is accounted for
- non-closing transport, gluing strain, or contradiction-like closure obstruction

Intended use:

- identify cells where defect is intrinsically compositional rather than merely local

## 5. Placement Rules

These diagnostics must be emitted on a separate surface from the standing report.

Allowed placements:

- per-sample diagnostic JSON or CSV
- per-cell aggregate summary
- separate markdown bridge report

Forbidden placements:

- insertion into `candidate_summary.csv`
- insertion into the standing headline table as if they were comparable metrics
- any scoreboard that implies they are new candidates

## 6. Minimum Output Scope

Implementation should stay minimal in the first pass.

The bridge should emit only:

- per-sample diagnostic rows
- per-cell aggregate summaries

That is enough to test the hypothesis without reopening the Gate8 court.

No new consumer family is required in the first pass.

The preferred first implementation is:

- supplemental output inside Gate8 execution
- written after the fixed standing path completes

## 7. Cell Expectations

These expectations must be fixed before reading results.

### 7.1 `clean_support`

Expected:

- high rotation may occur
- leakage should remain low
- closure defect should remain low

Interpretation:

- lawful contextual movement is allowed here
- defect-like escape is not

### 7.2 `surface_noisy_clean`

Expected:

- high rotation may occur
- leakage should remain low
- closure defect should remain low

Interpretation:

- surface wobble may move the local frame
- it should not look like actual defect

### 7.3 `direct_contradiction`

Expected:

- leakage or closure defect should rise
- simple lawful rotation alone should not be sufficient explanation

Interpretation:

- first-order contradiction should force more than benign reorientation

### 7.4 `distributed_incompatibility`

Expected:

- closure defect should rise especially strongly
- leakage may rise, but the strongest expectation is on non-closure

Interpretation:

- this is the cell most likely to expose transport-level failure rather than pointwise defect alone

## 8. Falsifiers

The bridge hypothesis weakens or fails under any of the following:

- `clean_support` shows ordinary high leakage
- `surface_noisy_clean` shows ordinary high leakage
- conflict cells do not raise `closure_defect` relative to clean/noisy cells
- `distributed_incompatibility` does not show stronger closure-style behavior than the clean cells
- diagnostics do not help explain why `gate7c` is strong on conflict cells and weak on Seam-tail behavior

More sharply:

- if Seam-tail spikes are not mostly high-rotation and low-leakage events, the rotation/leakage story weakens
- if conflict behavior does not separate into leakage or closure-defect elevation, the transition-first story weakens

## 9. Non-Promotion Rule

Even a successful bridge does not immediately earn:

- candidate promotion
- replacement of `F`
- replacement of `gate6f`
- a new mainline metric

What it earns, at most, is:

- justification for designing a later candidate that is more rotation-invariant and more leakage-sensitive

That later candidate would still need its own fixed-court test.

## 10. Failure Reading

If the bridge fails, the honest readings include:

- the Seam-tail burden is not mainly rotation/leakage confusion
- the current dynamic law is weak for another reason
- the current transport decomposition is not the right explanatory cut

Failure here does not license emergency metric proliferation.

## 11. Suggested Artifact Shape

First-pass artifact names may look like:

- `diagnostics/rotation_leakage_per_sample.csv`
- `diagnostics/rotation_leakage_by_cell.csv`
- `diagnostics/rotation_leakage_bridge_report.md`

The exact filenames are not sacred.

The separation rule is sacred:

- diagnostics must remain visibly separate from standing outputs

## 12. Exit Condition

This bridge phase is complete only if:

- the diagnostics are defined before implementation drift begins
- the four cell expectations are frozen up front
- the falsifiers are written before reading results
- the outputs stay diagnostic-only

Until then, this workstream is still doctrine without a proper bridge.
