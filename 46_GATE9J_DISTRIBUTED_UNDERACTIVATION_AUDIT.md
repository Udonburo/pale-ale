# Gate9J Distributed Underactivation Audit

Status: narrow blocker-audit spec, first implementation landed and first smoke execution recorded
Role: Gate9J support-anchor distributed-underactivation spec, not redesign settlement or operator opening
Date: 2026-03-21

Gate9J proceeds from:

- `44_GATE9I_SUPPORT_ANCHOR_CLEANER_CELL_DOMINANCE.md`
- `45_GATE9I_SUPPORT_ANCHOR_CLEANER_CELL_DOMINANCE_SMOKE.md`

The first Gate9J underactivation consumer now exists in:

- `tools/run_gate9j_distributed_underactivation_audit.py`

The first tracked Gate9J smoke execution read is now recorded in:

- `47_GATE9J_DISTRIBUTED_UNDERACTIVATION_SMOKE.md`

## 0. Why This Exists

Gate9I moved the blocker one layer inward.

What is now known is:

- support-anchor cleaner-cell dominance is still real
- the dominance is not explained by quietness noise
- the next named subblocker is `distributed_underactivation`

So the next honest move is not:

- another redesign candidate
- operator opening
- a return to abstract blocker language

It is:

- decomposing the distributed underactivation line itself

## 1. Scope

Gate9J studies only:

- the support-anchor rows from the Gate9H redesign line
- the conflict-side split between `direct_contradiction` and `distributed_incompatibility`
- the answer-target branches that sit inside the distributed cell

It does not:

- redesign the metric again
- reopen quietness analysis
- reopen operator admission

## 2. Public Question

The Gate9J question is:

- is support-anchor distributed underactivation family-wide, or is it concentrated on a narrower distributed branch

More concretely:

- does the distributed consistent-answer branch sit materially below the direct consistent baseline
- does the distributed cell split across answer-target branches
- is the gap loss better described as answer-coverage suppression than as token-only carryover

## 3. Public Object

Gate9J must emit:

- a support-anchor conflict-side registry
- per-cell and per-branch summaries
- an explicit comparison between the direct consistent baseline and the distributed consistent branch
- a deterministic blocker status payload

The public object is not rescue.

It is:

- a decomposition audit of the `distributed_underactivation` subblocker

## 4. Gate9J Falsifiers

Gate9J must keep these falsifiers explicit:

- distributed underactivation vanishes once the rows are branch-split
- the distributed cell does not actually split by answer target
- the gap loss is explainable as token-only carryover rather than answer-side suppression

If those fail, the subblocker is not yet properly named.

## 5. What This Audit Can Earn

At most, Gate9J can earn the right to say:

- `distributed_underactivation` is concentrated in a narrower branch with a specific compression pattern

It still does not earn:

- blocker resolution
- operator admission
- final redesign settlement

## 6. Current Memory Hook

The shortest acceptable sentence is:

- decompose distributed underactivation before trying to repair it
