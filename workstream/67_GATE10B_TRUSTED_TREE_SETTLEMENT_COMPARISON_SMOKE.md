# Gate10B Trusted-Tree Settlement Comparison Smoke

Status: tracked smoke execution snapshot
Role: tracked Gate10B settlement-comparison read, not Gate10 closeout, operator reopening, or broader trusted-tree settlement
Date: 2026-03-23

This first tracked Gate10B smoke read executes the settlement-comparison slice defined in:

- `66_GATE10B_TRUSTED_TREE_SETTLEMENT_COMPARISON.md`

The broader Gate10 court remains defined in:

- `63_GATE10_TRUSTED_TREE_SETTLEMENT_COURT.md`

The Gate10A eligibility gate that permitted this comparison remains recorded in:

- `64_GATE10A_TRUSTED_TREE_GENERALIZATION_ELIGIBILITY.md`
- `65_GATE10A_TRUSTED_TREE_GENERALIZATION_ELIGIBILITY_SMOKE.md`

## 0. Scope

This file records the first committed-code smoke execution of the Gate10B settlement-comparison slice.

It is not:

- Gate10 closeout
- operator reopening
- retroactive rewrite of Gate9
- broader trusted-tree settlement outside this declared slice

It is:

- a tracked handoff for the first Gate10B settlement-comparison slice
- a code-bound read on whether one eligible broader trusted-tree candidate earns slice-local settlement relative to the integrated forward-basis baseline
- the current scientific judgment on what Gate10B did and did not earn

The tracked evidence package is:

- `runs/gate10b_trusted_tree_settlement_comparison_smoke_from_gate10a/manifest.json`
- `runs/gate10b_trusted_tree_settlement_comparison_smoke_from_gate10a/trusted_tree_settlement_comparison_registry.jsonl`
- `runs/gate10b_trusted_tree_settlement_comparison_smoke_from_gate10a/trusted_tree_settlement_comparison_policy_compare.csv`
- `runs/gate10b_trusted_tree_settlement_comparison_smoke_from_gate10a/trusted_tree_settlement_comparison_status.json`
- `runs/gate10b_trusted_tree_settlement_comparison_smoke_from_gate10a/gate10b_trusted_tree_settlement_comparison_read.md`

## 1. Source And Bind

This smoke run consumes the first Gate10A eligibility bundle:

- `source_gate10a_run_id = gate10a_trusted_tree_generalization_eligibility_smoke_from_gate9q`
- `source_gate10a_code_git_commit = b1092e05f453dfa1a0926d5001d91cd9badbab32`

The upstream Gate9Q source remains:

- `source_gate9q_run_id = gate9q_post_adoption_integration_smoke_from_gate9p`
- `source_gate9q_code_git_commit = 96b10238f126f6d89ee845125835baef58a5d632`

The Gate10B comparison bind is:

- `method_id = gate10b_trusted_tree_settlement_comparison_v1`
- `code_git_commit = 54e31e4bb0398f0844c1c2baeb923d32e1a13bec`

## 2. What Landed

Gate10B asks only:

- whether one eligible broader trusted-tree candidate survives narrow comparison against the integrated forward-basis baseline
- whether conflict-side bridge preservation remains clear
- whether decision-relevant gain beyond baseline is present
- whether the result stays inside the Gate10 non-promotion boundary

It remains a slice-local comparison only.

## 3. Smoke Read

### 3.1 The Preserved Baseline Remains Intact

The baseline guard statuses are:

- `forward_basis_baseline_preservation_status = clear`
- `non_retroactive_memory_preservation_status = clear`
- `operator_admission_still_denied_status = confirmed`

The comparison registry matches that read:

- cleaner `8` edges remain in the `adopted_split_baseline` lane
- all `8` are still role-changed forward-basis edges
- no baseline row pressures retroactive reinterpretation or operator reopening

So Gate10B does not relitigate the integrated split.

It compares against it as preserved baseline.

### 3.2 The Broader Candidate Preserves Conflict-Side Bridge

The candidate-side preservation statuses are:

- `conflict_side_bridge_preservation_status = clear`
- `trusted_tree_semantics_broadening_pressure_status = clear`
- `operator_adjacent_rescue_pressure_status = clear`

The policy summary matches that read:

- conflict `8` edges define the `broader_candidate_opening_lane`
- all `8` remain forward-basis preserved without role change
- no silent broadening or rescue pressure appears in the declared slice

This matters because the broader candidate is not buying settlement by hiding anomaly-side structure.

### 3.3 The Comparison Earns Slice-Local Settlement Only

The outcome statuses are:

- `decision_relevant_gain_beyond_baseline_status = present`
- `comparison_outcome_status = settled`
- `broader_tree_settlement_non_promotion_status = clear`
- `next_named_blocker = ""`

So Gate10B does not say the broader trusted-tree line is globally settled.

It says only:

- this declared comparison slice earns `settled`

## 4. Current Scientific Judgment

The correct Gate10B smoke judgment is:

- Gate10B succeeded as a narrow settlement-comparison slice
- the eligible broader candidate survives comparison against the integrated forward-basis baseline
- conflict-side bridge preservation remains clear
- decision-relevant gain beyond baseline is present
- the result remains slice-local and non-promotional
- operator admission remains denied

The strongest honest sentence is:

- `Gate10B shows that one eligible broader trusted-tree candidate earns slice-local settlement relative to the integrated forward-basis baseline, while operator admission remains denied and broader trusted-tree settlement outside this declared comparison remains unearned.`

## 5. What This Smoke Run Earns

This smoke run earns the right to say:

- the first Gate10B declared comparison slice is `settled`
- Gate10 may proceed from eligibility-only language to a slice-local settlement sentence
- no named blocker currently survives inside this declared comparison slice

## 6. What This Smoke Run Does Not Earn

This smoke run does not earn the right to say:

- Gate10 as a whole is settled
- operator admission should reopen
- earlier Gate9 memory should be retroactively rewritten
- broader trusted-tree settlement outside this slice is now resolved

## 7. Next Honest Move

The next honest move is not:

- declare full Gate10 settlement
- reopen operator admission
- rewrite earlier Gate9 reads

The next honest move is:

- decide whether Gate10 should open another narrow settlement slice or begin closeout-level Gate10 memory work
