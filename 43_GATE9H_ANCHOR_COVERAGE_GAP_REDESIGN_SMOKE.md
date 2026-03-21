# Gate9H Anchor-Coverage-Gap Redesign Smoke

Status: tracked smoke execution snapshot
Role: tracked Gate9H redesign-candidate read, not operator opening or final redesign settlement
Date: 2026-03-21

This first tracked Gate9H smoke read executes the narrow redesign audit defined in:

- `42_GATE9H_ANCHOR_COVERAGE_GAP_REDESIGN.md`

## 0. Scope

This file records the first committed-code smoke execution of the Gate9H anchor-coverage-gap redesign audit.

It is not:

- final anchor redesign settlement
- operator opening
- a new anchor lane
- a new closure convention

It is:

- a tracked handoff for the smallest redesign candidate after Gate9G collapse
- a code-bound read on whether the redesign candidate escapes triviality
- the current scientific judgment on whether the next blocker is now cleaner-cell dominance

The tracked evidence package is:

- `runs/gate9h_anchor_coverage_gap_redesign_smoke_from_gate9g/manifest.json`
- `runs/gate9h_anchor_coverage_gap_redesign_smoke_from_gate9g/anchor_coverage_gap_redesign_registry.jsonl`
- `runs/gate9h_anchor_coverage_gap_redesign_smoke_from_gate9g/anchor_coverage_gap_redesign_by_cell_anchor.csv`
- `runs/gate9h_anchor_coverage_gap_redesign_smoke_from_gate9g/anchor_coverage_gap_redesign_status.json`
- `runs/gate9h_anchor_coverage_gap_redesign_smoke_from_gate9g/gate9h_anchor_coverage_gap_redesign_read.md`

## 1. Source And Bind

This smoke run consumes the first Gate9G triviality bundle:

- `source_gate9g_run_id = gate9g_anchor_conditioned_triviality_smoke_from_gate9f`
- `source_gate9g_code_git_commit = 6586002abc9dbb773f66015e7a15a88702e3ed69`

The Gate9H redesign bind is:

- `method_id = gate9h_anchor_coverage_gap_redesign_audit_v1`
- `code_git_commit = 80f0a3c4916982fd5b130a96670c656068401157`

The redesign candidate is:

- `anchor_coverage_gap_abs_v1`

## 2. What Landed

Gate9H now replaces the collapsed closure-defect read with one minimal candidate:

- `coverage_gap_abs := |anchor_answer_coverage - anchor_token_coverage|`

Nothing else changes.

The same bundle remains in force.

The same anchor-conditioned rows remain in force.

The question is only whether this candidate escapes triviality cleanly enough to move the blocker.

## 3. Smoke Read

### 3.1 Non-Triviality

The status payload is:

- `redesign_candidate_nontriviality_status = provisionally_clear`
- `n_nontrivial_gap_candidate_rows = 12`
- `n_collapsed_gap_candidate_rows = 0`

So the minimal redesign candidate does clear the specific Gate9G blocker.

It does not collapse numerically.

### 3.2 Conflict Availability

The redesign candidate remains available on the conflict-side rows:

- `conflict_anchor_availability_status = clear`
- `conflict_direct_mean_gap = 0.112206`
- `conflict_distributed_mean_gap = 0.114176`

So the redesign candidate is not blocked by missingness anymore.

### 3.3 Cleaner-Cell Dominance

The decisive read is:

- `support_anchor_cleaner_dominance_status = triggered`
- `support_cleaner_max_mean_gap = 0.150462`
- `support_conflict_max_mean_gap = 0.136027`

This is the correct next blocker.

The redesign candidate escapes triviality, but it does not escape cleaner-cell dominance.

### 3.4 Admission Readiness

The final Gate9H redesign status is:

- `redesign_admission_readiness_status = denied`
- `next_named_blocker = cleaner_cell_dominance`

So Gate9H does not earn admission.

It earns a narrower sentence:

- the triviality blocker can be escaped by a minimal redesign candidate
- cleaner-cell dominance now becomes the active named blocker on that redesign line

## 4. Current Scientific Judgment

The correct Gate9H smoke judgment is:

- Gate9H succeeded as a redesign-candidate audit
- the `coverage_gap_abs` candidate clears the Gate9G triviality blocker
- the redesign line remains denied because cleaner-cell dominance still triggers on the support anchor family

The strongest honest sentence is:

- `Gate9H shows that anchor-conditioned triviality can be escaped without changing the law, but the redesign candidate immediately hands the line to cleaner-cell dominance rather than operator admission.`

## 5. What This Smoke Run Earns

This smoke run earns the right to say:

- the next blocker is no longer anchor-conditioned triviality
- the next blocker is now explicitly `cleaner_cell_dominance`
- future work can attack that blocker directly without pretending anchor-conditioned redesign is still undefined

## 6. What This Smoke Run Does Not Earn

This smoke run does not earn the right to say:

- the redesign is final
- operator admission should open
- the support cycle is now conflict-led

## 7. Next Honest Move

The next honest move is not:

- operator opening
- final redesign branding
- court inflation

The next honest move is:

- attack `cleaner_cell_dominance` as the next named blocker on the redesign line
