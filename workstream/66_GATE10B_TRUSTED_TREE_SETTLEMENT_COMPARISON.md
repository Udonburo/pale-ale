# Gate10B Trusted-Tree Settlement Comparison

Status: first implementation landed and first smoke execution recorded
Role: Gate10B settlement-comparison slice, not Gate10 closeout, operator chapter, or retroactive rewrite
Date: 2026-03-23

Gate10B proceeds from:

- `63_GATE10_TRUSTED_TREE_SETTLEMENT_COURT.md`
- `64_GATE10A_TRUSTED_TREE_GENERALIZATION_ELIGIBILITY.md`
- `65_GATE10A_TRUSTED_TREE_GENERALIZATION_ELIGIBILITY_SMOKE.md`

The first Gate10B settlement-comparison consumer now exists in:

- `tools/run_gate10b_trusted_tree_settlement_comparison.py`

The first tracked Gate10B settlement-comparison smoke read is now recorded in:

- `67_GATE10B_TRUSTED_TREE_SETTLEMENT_COMPARISON_SMOKE.md`

The second Gate10C settlement-comparison slice is now tracked in:

- `68_GATE10C_SECOND_SETTLEMENT_COMPARISON.md`

## 0. Scope

Gate10B is the first narrow settlement-comparison slice under the Gate10 court.

Gate10B does:

- compare an eligible broader trusted-tree candidate against the Gate9Q integrated forward-basis baseline
- ask whether that broader candidate earns doctrine-safe settlement under the declared Gate10 court
- preserve the adopted split as baseline rather than reopening it

Gate10B does not:

- reopen operator admission
- reopen graph-wide operator design
- retroactively reinterpret Gate9 reads
- replace the integrated Gate9N/P/Q split
- settle the entire Gate10 workstream by one comparison alone
- broaden trusted-tree semantics by convenience
- reopen scalar rescue

## 1. Public Question

The Gate10B question is:

- `can an eligible broader trusted-tree candidate earn doctrine-safe settlement relative to the integrated forward-basis baseline, without degrading conflict-side bridge preservation or violating the Gate10 boundary?`

This is narrower than:

- final Gate10 closeout
- broad trusted-tree victory by declaration
- any operator-adjacent reopening

It is wider than:

- Gate10A eligibility alone

## 2. Comparison Lanes

Gate10B compares exactly two lanes.

### 2.1 Baseline Lane

The baseline lane is:

- the Gate9Q integrated forward-basis split

That baseline remains:

- preserved
- forward-only
- non-retroactive

### 2.2 Broader Candidate Lane

The broader candidate lane is:

- one declared broader trusted-tree / residual-chord candidate that has already passed Gate10A eligibility

Gate10B may compare only one declared broader candidate at a time.

This keeps the settlement court narrow and falsifiable.

## 3. Required Settlement Checks

The broader candidate may earn settlement only if all of the following remain clear.

### 3.1 Forward-Basis Baseline Preservation

Settlement is denied if:

- the integrated forward-basis split stops functioning as preserved baseline
- the comparison silently turns into relitigation of the adopted split

### 3.2 Conflict-Side Bridge Preservation

Settlement is denied if:

- broader trusted-tree gain survives only by degrading conflict-side bridge legibility
- anomaly-side residual burden becomes less legible than under the preserved baseline

Gate10B is not allowed to buy cleaner-side improvement by hiding conflict-side structure.

### 3.3 Non-Retroactive Memory Preservation

Settlement is denied if:

- broader comparison requires earlier Gate9 reads to be reclassified, resettled, or silently re-explained

### 3.4 No Operator-Adjacent Rescue

Settlement is denied if:

- broader standing can be maintained only by leaning toward operator-like smoothing, field-style rescue, or graph-wide behavior

### 3.5 No Silent Broadening Of Tree Semantics

Settlement is denied if:

- broader candidate success depends on undeclared expansion of trusted-tree meaning
- new semantics are introduced through carve-out, role broadening, or bundle-specific exception logic

### 3.6 Decision-Relevant Gain Beyond Baseline

Settlement is denied if:

- the broader candidate does not produce decision-relevant gain beyond the integrated forward-basis baseline
- the comparison merely renames the same effective result without improving the declared burden

Gate10B does not require a breakthrough metric.

It does require:

- a court-relevant gain that matters relative to baseline

## 4. Falsifiers

The broader candidate fails Gate10B settlement if any of the following happens:

- forward-basis baseline preservation fails
- conflict-side bridge degrades
- retroactive reinterpretation pressure appears
- operator-adjacent rescue pressure appears
- trusted-tree semantics broaden silently
- decision-relevant gain beyond baseline is absent
- scalar masking is needed to preserve the desired verdict

## 5. Output Ladder For This Slice

Gate10B uses the Gate10 outcome ladder, but only for this declared comparison slice.

### 5.1 Settled

Use `settled` only if:

- the broader candidate survives comparison against baseline
- no falsifier fires
- a decision-relevant gain beyond baseline is present

### 5.2 Bounded Keep

Use `bounded keep` if:

- the broader candidate remains informative or practically useful relative to baseline
- but does not earn doctrine-safe settlement
- and the integrated forward-basis baseline remains the keepable mainline reference

### 5.3 Denied

Use `denied` if:

- the broader candidate fails the declared settlement comparison
- and should not be casually reopened without a new named blocker or new constitutional reason

### 5.4 Deferred

Use `deferred` if:

- the comparison remains too incomplete for settlement
- but denial is not yet justified

## 6. Forbidden

The following remain forbidden in Gate10B:

- no operator reopening
- no graph-wide operator language as rescue
- no retroactive rewrite of Gate9
- no silent broadening of trusted-tree semantics
- no scalar masking
- no KAGAMI rhetoric
- no sheaf or higher-gauge branding
- no benchmark-zoo expansion

## 7. Non-Promotion Clause

Even a positive Gate10B result does not immediately earn:

- operator admission
- full Gate10 closeout
- broader trusted-tree settlement outside the declared comparison slice
- rewrite of earlier Gate9 memory

Gate10B may earn only:

- a slice-local settlement sentence
- a bounded-keep sentence
- a denial sentence
- or a defer sentence

## 8. Minimal Implementation Obligations

Any Gate10B implementation must preserve:

- the Gate9Q integrated baseline
- forward-basis adoption as already integrated
- operator admission still denied
- non-retroactive Gate9 memory

Any implementation must emit explicit status for:

- forward-basis-baseline preservation
- conflict-side-bridge preservation
- non-retroactive-memory preservation
- operator-adjacent-rescue pressure
- trusted-tree-semantics broadening pressure
- decision-relevant gain beyond baseline
- comparison outcome
- next-named-blocker

## 9. What Gate10B Is Not

Gate10B is not:

- Gate10 closeout
- a graph-wide operator retry
- a trusted-tree slogan chapter
- a rewrite of Gate9

Gate10B is:

- the first narrow settlement-comparison slice for a broader trusted-tree candidate under the preserved Gate9Q baseline

## 10. Memory Hook

The Gate10B sentence is:

- Gate10B compares one eligible broader trusted-tree candidate against the integrated forward-basis baseline under the declared settlement court

The shortest acceptable memory hook is:

- `comparison before settlement claim`
