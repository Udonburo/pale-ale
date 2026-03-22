# Gate10D Third Trusted-Tree Settlement Comparison Smoke

Status: tracked smoke execution snapshot
Role: tracked Gate10D third settlement-comparison read, not Gate10 closeout, operator reopening, or broader trusted-tree settlement
Date: 2026-03-23

This first tracked Gate10D smoke read executes the third settlement-comparison slice defined in:

- `70_GATE10D_THIRD_SETTLEMENT_COMPARISON.md`

The broader Gate10 court remains defined in:

- `63_GATE10_TRUSTED_TREE_SETTLEMENT_COURT.md`

The prior settled slices that Gate10D preserves remain recorded in:

- `66_GATE10B_TRUSTED_TREE_SETTLEMENT_COMPARISON.md`
- `67_GATE10B_TRUSTED_TREE_SETTLEMENT_COMPARISON_SMOKE.md`
- `68_GATE10C_SECOND_SETTLEMENT_COMPARISON.md`
- `69_GATE10C_SECOND_SETTLEMENT_COMPARISON_SMOKE.md`

The next Gate10E interim broader-judgment / pre-closeout memory slice is now tracked in:

- `72_GATE10E_INTERIM_BROADER_JUDGMENT.md`

## 0. Scope

This file records the first committed-code smoke execution of the Gate10D third settlement-comparison slice.

It is not:

- Gate10 closeout
- operator reopening
- retroactive rewrite of Gate9 or Gate10A/B/C
- broader trusted-tree settlement outside the declared first, second, and third slices

It is:

- a tracked handoff for the first Gate10D third settlement-comparison slice
- a code-bound read on whether the declared `direct_contradiction` third candidate earns slice-local settlement relative to the integrated forward-basis baseline
- the current scientific judgment on what Gate10D did and did not earn

The tracked evidence package is:

- `runs/gate10d_trusted_tree_third_settlement_comparison_smoke_from_gate10c/manifest.json`
- `runs/gate10d_trusted_tree_third_settlement_comparison_smoke_from_gate10c/trusted_tree_third_settlement_comparison_registry.jsonl`
- `runs/gate10d_trusted_tree_third_settlement_comparison_smoke_from_gate10c/trusted_tree_third_settlement_comparison_policy_compare.csv`
- `runs/gate10d_trusted_tree_third_settlement_comparison_smoke_from_gate10c/trusted_tree_third_settlement_comparison_status.json`
- `runs/gate10d_trusted_tree_third_settlement_comparison_smoke_from_gate10c/gate10d_trusted_tree_third_settlement_comparison_read.md`

## 1. Source And Bind

This smoke run consumes the first Gate10C settlement-comparison bundle:

- `source_gate10c_run_id = gate10c_trusted_tree_second_settlement_comparison_smoke_from_gate10b`
- `source_gate10c_code_git_commit = 3c7cfa8ec9d726d7fcd8984be661dc14b2863860`

The frozen upstream Gate10B source resolved through the Gate10C manifest is:

- `source_gate10b_run_id = gate10b_trusted_tree_settlement_comparison_smoke_from_gate10a`
- `source_gate10b_code_git_commit = 54e31e4bb0398f0844c1c2baeb923d32e1a13bec`

The deeper upstream sources remain:

- `source_gate10a_run_id = gate10a_trusted_tree_generalization_eligibility_smoke_from_gate9q`
- `source_gate10a_code_git_commit = b1092e05f453dfa1a0926d5001d91cd9badbab32`
- `source_gate9q_run_id = gate9q_post_adoption_integration_smoke_from_gate9p`
- `source_gate9q_code_git_commit = 96b10238f126f6d89ee845125835baef58a5d632`

The Gate10D comparison bind is:

- `method_id = gate10d_third_settlement_comparison_v1`
- `code_git_commit = ead026531b85661f61947c3f5241443291e886ea`

## 2. What Landed

Gate10D asks only:

- whether the declared `direct_contradiction` third candidate survives narrow comparison against the same integrated forward-basis baseline
- whether the first Gate10B and Gate10C settled slices remain preserved
- whether conflict-side bridge preservation remains clear
- whether the result stays inside the Gate10 non-promotion boundary

It remains a third slice-local comparison only.

## 3. Smoke Read

### 3.1 The Preserved Baseline And Prior Slices Remain Intact

The baseline and carry-forward guard statuses are:

- `forward_basis_baseline_preservation_status = clear`
- `gate10b_slice_non_retroactive_preservation_status = clear`
- `gate10c_slice_non_retroactive_preservation_status = clear`
- `non_retroactive_memory_preservation_status = clear`
- `operator_admission_still_denied_status = confirmed`

The comparison registry matches that read:

- cleaner `8` edges remain the `adopted_split_baseline` lane
- neither the first nor second settled slice is weakened or reinterpreted
- no row pressures retroactive rewrite or operator reopening

So Gate10D does not relitigate the integrated split or the prior two settled slices.

### 3.2 The Third Candidate Is Declarative And Conflict-Side Preserving

The third-candidate statuses are:

- `third_candidate_declaration_status = clear`
- `conflict_side_bridge_preservation_status = clear`
- `trusted_tree_semantics_broadening_pressure_status = clear`
- `operator_adjacent_rescue_pressure_status = clear`

The policy summary matches that read:

- conflict `4` edges define the `direct_contradiction` third-candidate lane
- all `4` are declaratively extracted from the Gate10B broader candidate opening lane through the Gate10C manifest-resolved source rule
- no silent broadening or rescue pressure appears in the declared slice

This matters because the third slice is not manufactured by hidden branching or doctrine drift.

### 3.3 The Third Comparison Also Earns Slice-Local Settlement

The outcome statuses are:

- `decision_relevant_gain_beyond_baseline_status = present`
- `comparison_outcome_status = settled`
- `broader_tree_settlement_non_promotion_status = clear`
- `next_named_blocker = ""`

So Gate10D does not say the broader trusted-tree line is globally settled.

It says only:

- this declared third comparison slice also earns `settled`

## 4. Current Scientific Judgment

The correct Gate10D smoke judgment is:

- Gate10D succeeded as a third narrow settlement-comparison slice
- the declared `direct_contradiction` third candidate survives comparison against the integrated forward-basis baseline
- the first Gate10B and Gate10C settled slices remain preserved
- conflict-side bridge preservation remains clear
- decision-relevant gain beyond baseline is present
- the result remains slice-local and non-promotional
- operator admission remains denied

The strongest honest sentence is:

- `Gate10D shows that the declared direct_contradiction third candidate also earns slice-local settlement relative to the integrated forward-basis baseline, while operator admission remains denied and broader trusted-tree settlement outside the declared first, second, and third slices remains unearned.`

## 5. What This Smoke Run Earns

This smoke run earns the right to say:

- the declared Gate10D third comparison slice is `settled`
- three declared narrow slices have now earned slice-local `settled` under the preserved Gate10 court
- no named blocker currently survives inside this declared third comparison slice

## 6. What This Smoke Run Does Not Earn

This smoke run does not earn the right to say:

- Gate10 as a whole is settled
- operator admission should reopen
- earlier Gate9 or Gate10A/B/C memory should be retroactively rewritten
- broader trusted-tree settlement outside the declared first, second, and third slices is now resolved

## 7. Next Honest Move

The next honest move is not:

- declare full Gate10 settlement
- reopen operator admission
- rewrite earlier Gate9 or Gate10A/B/C reads

The next honest move is:

- open an interim broader-judgment / pre-closeout memory slice before any Gate10 closeout claim
