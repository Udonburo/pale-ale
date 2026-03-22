# Gate10D Third Trusted-Tree Settlement Comparison

Status: first implementation landed and first smoke execution recorded
Role: third narrow settlement-comparison slice, not Gate10 closeout, operator chapter, or retroactive rewrite
Date: 2026-03-23

Gate10D proceeds from:

- `63_GATE10_TRUSTED_TREE_SETTLEMENT_COURT.md`
- `68_GATE10C_SECOND_SETTLEMENT_COMPARISON.md`
- `69_GATE10C_SECOND_SETTLEMENT_COMPARISON_SMOKE.md`

The first Gate10D third settlement-comparison consumer now exists in:

- `tools/run_gate10d_third_settlement_comparison.py`

The first tracked Gate10D third settlement-comparison smoke read is now recorded in:

- `71_GATE10D_THIRD_SETTLEMENT_COMPARISON_SMOKE.md`

## 0. Scope

Gate10D is the third narrow settlement-comparison slice under the Gate10 court.

Gate10D does:

- compare one third declared broader trusted-tree candidate against the same Gate9Q integrated forward-basis baseline used in Gate10B and Gate10C
- ask whether that third candidate also earns slice-local settlement under the declared Gate10 court
- test whether the first two slice-local `settled` results are merely local accidents or part of a minimally repeatable broader pattern

Gate10D does not:

- reopen operator admission
- reopen graph-wide operator design
- retroactively reinterpret Gate9 or Gate10A/B/C
- replace or weaken the already recorded Gate10B or Gate10C slice-local settlements
- settle the entire Gate10 workstream by three slices alone
- broaden trusted-tree semantics by convenience
- reopen scalar rescue

## 1. Frozen Source Lineage

The controlling Gate10D source run is:

- `runs/gate10c_trusted_tree_second_settlement_comparison_smoke_from_gate10b`

Gate10D must preserve the baseline and prior slice judgments recorded there.

The third-candidate extraction source is not invented locally.

It is:

- the `source_gate10b_dir` explicitly recorded in the Gate10C manifest

For this slice, that means the implementation may walk one step upstream to the frozen Gate10B bundle already named by Gate10C.

### 1.1 Preserved Baseline Lane

The baseline lane remains:

- the same Gate9Q integrated forward-basis split preserved in Gate10B and Gate10C

Gate10D is not allowed to change the baseline definition.

### 1.2 Third Declared Broader Candidate

The third candidate lane is:

- the `direct_contradiction` sublane extracted declaratively from the Gate10B broader candidate opening lane referenced by the Gate10C manifest

For Gate10D, `direct_contradiction` means:

- rows emitted from the referenced Gate10B source context whose broader candidate lineage remains conflict-side
- and whose `cell_id` is `direct_contradiction`

No other third-candidate declaration is in scope for this slice.

## 2. Public Question

The Gate10D question is:

- `does the declared direct_contradiction third broader candidate also earn slice-local settlement relative to the same integrated forward-basis baseline, without degrading the preserved Gate10 boundary?`

This is narrower than:

- broader Gate10 settlement
- trusted-tree victory by repetition slogan
- any operator-adjacent reopening

It is wider than:

- Gate10B and Gate10C alone

## 3. Why Gate10D Exists

Gate10B earned:

- one declared slice-local `settled`

Gate10C earned:

- one second declared slice-local `settled`

But Gate10B/C still do not earn:

- broader Gate10 settlement
- operator reopening
- retroactive reinterpretation

So the next honest move is not:

- declare broader settlement from two slices alone

It is:

- ask whether the remaining declared third candidate also survives narrow settlement comparison under the same preserved baseline

## 4. Required Settlement Checks

The third candidate may earn slice-local settlement only if all of the following remain clear.

### 4.1 Forward-Basis Baseline Preservation

Settlement is denied if:

- the Gate9Q integrated forward-basis split stops functioning as preserved baseline
- the third comparison silently turns into relitigation of the adopted split

### 4.2 Gate10B Slice Non-Retroactive Preservation

Settlement is denied if:

- Gate10D requires the first Gate10B settled slice to be weakened, reclassified, or silently reinterpreted
- the third slice succeeds only by undoing what Gate10B already earned

### 4.3 Gate10C Slice Non-Retroactive Preservation

Settlement is denied if:

- Gate10D requires the second Gate10C settled slice to be weakened, reclassified, or silently reinterpreted
- the third slice succeeds only by undoing what Gate10C already earned

### 4.4 Third-Candidate Declaration Integrity

Settlement is denied if:

- the `direct_contradiction` third candidate cannot be extracted declaratively from the frozen source lineage
- the extraction requires undeclared role surgery, hidden branching, or bundle-specific carve-out logic

### 4.5 Conflict-Side Bridge Preservation

Settlement is denied if:

- the third candidate survives only by degrading conflict-side bridge legibility
- anomaly-side residual burden becomes less legible than under the preserved baseline

### 4.6 Non-Retroactive Memory Preservation

Settlement is denied if:

- the third comparison requires earlier Gate9 or Gate10A/B/C reads to be rewritten, re-explained, or silently reclassified

### 4.7 No Operator-Adjacent Rescue

Settlement is denied if:

- the third candidate can be maintained only by leaning toward operator-like smoothing, field-style rescue, or graph-wide behavior

### 4.8 No Silent Broadening Of Tree Semantics

Settlement is denied if:

- the third candidate succeeds only under undeclared expansion of trusted-tree meaning
- new semantics are introduced through carve-out, role broadening, or exception logic

### 4.9 Decision-Relevant Gain Beyond Baseline

Settlement is denied if:

- the third candidate does not produce decision-relevant gain beyond the integrated forward-basis baseline
- the comparison merely renames baseline-relative burden without improving the declared read

Gate10D does not require a breakthrough metric.

It does require:

- a court-relevant gain that matters relative to baseline

## 5. Falsifiers

The third candidate fails Gate10D settlement if any of the following happens:

- forward-basis baseline preservation fails
- Gate10B settled slice must be reinterpreted
- Gate10C settled slice must be reinterpreted
- third-candidate declaration integrity fails
- conflict-side bridge degrades
- retroactive reinterpretation pressure appears
- operator-adjacent rescue pressure appears
- trusted-tree semantics broaden silently
- decision-relevant gain beyond baseline is absent
- scalar masking is needed to preserve the desired verdict

## 6. Output Ladder For This Slice

Gate10D uses the Gate10 outcome ladder, but only for this declared third comparison slice.

### 6.1 Settled

Use `settled` only if:

- the third candidate survives comparison against baseline
- no falsifier fires
- a decision-relevant gain beyond baseline is present

### 6.2 Bounded Keep

Use `bounded keep` if:

- the third candidate remains informative or practically useful relative to baseline
- but does not earn doctrine-safe settlement
- the integrated forward-basis baseline remains the keepable mainline reference
- and the first Gate10B and Gate10C settled slices remain preserved as already recorded

### 6.3 Denied

Use `denied` if:

- the third candidate fails the declared settlement comparison
- and should not be casually reopened without a new named blocker or new constitutional reason

### 6.4 Deferred

Use `deferred` if:

- the comparison remains too incomplete for settlement
- but denial is not yet justified

## 7. Expected Outputs

Any Gate10D implementation must emit exactly these files:

- `manifest.json`
- `checksums.json`
- `trusted_tree_third_settlement_comparison_registry.jsonl`
- `trusted_tree_third_settlement_comparison_policy_compare.csv`
- `trusted_tree_third_settlement_comparison_status.json`
- `gate10d_trusted_tree_third_settlement_comparison_read.md`

## 8. Required Status Keys

Any Gate10D implementation must emit explicit status for:

- `forward_basis_baseline_preservation_status`
- `gate10b_slice_non_retroactive_preservation_status`
- `gate10c_slice_non_retroactive_preservation_status`
- `third_candidate_declaration_status`
- `conflict_side_bridge_preservation_status`
- `non_retroactive_memory_preservation_status`
- `operator_adjacent_rescue_pressure_status`
- `trusted_tree_semantics_broadening_pressure_status`
- `decision_relevant_gain_beyond_baseline_status`
- `comparison_outcome_status`
- `operator_admission_still_denied_status`
- `broader_tree_settlement_non_promotion_status`
- `next_named_blocker`

## 9. Forbidden

The following remain forbidden in Gate10D:

- no operator reopening
- no graph-wide operator language as rescue
- no retroactive rewrite of Gate9 or Gate10A/B/C
- no silent broadening of trusted-tree semantics
- no scalar masking
- no KAGAMI rhetoric
- no sheaf or higher-gauge branding
- no benchmark-zoo expansion
- no third-candidate declaration other than the in-scope `direct_contradiction` extraction

## 10. Non-Promotion Clause

Even a positive Gate10D result does not immediately earn:

- operator admission
- full Gate10 closeout
- broader trusted-tree settlement outside the declared first, second, and third comparison slices
- rewrite of earlier Gate9 or Gate10 memory

Gate10D may earn only:

- a third slice-local settlement sentence
- a bounded-keep sentence
- a denial sentence
- or a defer sentence

## 11. Delegation Boundary

This file is safe to hand to an implementation worker only under frozen-spec discipline.

Allowed work:

- narrow consumer implementation
- unit tests
- status payload emission
- tracked smoke handoff formatting

Not delegated:

- blocker naming
- falsifier redesign
- scope widening
- doctrine changes
- operator reopening
- retroactive reinterpretation of Gate9 or Gate10A/B/C

If the spec appears insufficient, implementation must stop and report the gap instead of inventing behavior.

## 12. Memory Hook

The Gate10D sentence is:

- Gate10D compares one third declared broader trusted-tree candidate against the same integrated forward-basis baseline under the preserved Gate10 boundary

The shortest acceptable memory hook is:

- `third slice before broader claim`
