# Gate9I Support-Anchor Cleaner-Cell Dominance

Status: narrow blocker-audit spec, first implementation landed and first smoke execution recorded
Role: Gate9I support-anchor dominance blocker spec, not redesign settlement or operator opening
Date: 2026-03-21

Gate9I proceeds from:

- `42_GATE9H_ANCHOR_COVERAGE_GAP_REDESIGN.md`
- `43_GATE9H_ANCHOR_COVERAGE_GAP_REDESIGN_SMOKE.md`

The first Gate9I dominance consumer now exists in:

- `tools/run_gate9i_support_anchor_cleaner_cell_dominance_audit.py`

The first tracked Gate9I smoke execution read is now recorded in:

- `45_GATE9I_SUPPORT_ANCHOR_CLEANER_CELL_DOMINANCE_SMOKE.md`

The next distributed-underactivation slice is now recorded in:

- `46_GATE9J_DISTRIBUTED_UNDERACTIVATION_AUDIT.md`
- `47_GATE9J_DISTRIBUTED_UNDERACTIVATION_SMOKE.md`

## 0. Why This Exists

Gate9H moved the blocker cleanly.

What is now known is:

- the minimal redesign candidate escapes triviality
- conflict-side availability remains clear
- admission still stays denied
- the active blocker is now `cleaner_cell_dominance`

So the next honest move is not:

- operator opening
- final redesign branding
- broad conflict rescue

It is:

- determining what the support-anchor cleaner-cell dominance is actually made of

## 1. Scope

Gate9I studies only:

- the support-anchor rows from the Gate9H redesign line
- quietness-paired cleaner rows
- the support-side conflict split between `direct_contradiction` and `distributed_incompatibility`

It does not:

- redesign the metric again
- touch conflict-anchor redesign
- reopen operator admission

## 2. Public Question

The Gate9I question is:

- is support-anchor cleaner-cell dominance primarily a quietness artifact, or is it a cleaner plateau plus conflict-side underactivation

More concretely:

- does `surface_noisy_clean` still corroborate the cleaner-side support level
- does `distributed_incompatibility` lag behind `direct_contradiction`
- can quietness noise alone explain the current dominance line

## 3. Public Object

Gate9I must emit:

- support-anchor registry with quietness-pair metadata
- per-cell support means
- per-quietness-pair deltas
- a deterministic blocker status payload

The public object is not rescue.

It is:

- a decomposition audit of the current support-anchor dominance blocker

## 4. Gate9I Falsifiers

Gate9I must keep these falsifiers explicit:

- dominance disappears once noisy cleaner rows are considered
- dominance is fully explainable as quietness noise
- there is no conflict-side split between `direct_contradiction` and `distributed_incompatibility`

If those fail, the blocker is not yet properly named.

## 5. What This Audit Can Earn

At most, Gate9I can earn the right to say:

- the support-anchor dominance blocker is real and decomposes into a specific source pattern

It still does not earn:

- blocker resolution
- operator admission
- final redesign settlement

## 6. Current Memory Hook

The shortest acceptable sentence is:

- decompose cleaner-cell dominance before trying to defeat it
