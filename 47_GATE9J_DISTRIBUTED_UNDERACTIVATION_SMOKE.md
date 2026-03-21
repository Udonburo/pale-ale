# Gate9J Distributed Underactivation Smoke

Status: tracked smoke execution snapshot
Role: tracked Gate9J distributed-underactivation read, not blocker resolution or operator opening
Date: 2026-03-21

This first tracked Gate9J smoke read executes the narrow underactivation audit defined in:

- `46_GATE9J_DISTRIBUTED_UNDERACTIVATION_AUDIT.md`

## 0. Scope

This file records the first committed-code smoke execution of the Gate9J distributed-underactivation audit.

It is not:

- blocker resolution
- final redesign settlement
- operator opening
- a new metric family

It is:

- a tracked handoff for the next named subblocker after Gate9I
- a code-bound read on whether distributed underactivation is family-wide or branch-concentrated
- the current scientific judgment on what sits inside the support-anchor conflict-side split

The tracked evidence package is:

- `runs/gate9j_distributed_underactivation_smoke_from_gate9i/manifest.json`
- `runs/gate9j_distributed_underactivation_smoke_from_gate9i/distributed_underactivation_registry.jsonl`
- `runs/gate9j_distributed_underactivation_smoke_from_gate9i/distributed_underactivation_by_cell.csv`
- `runs/gate9j_distributed_underactivation_smoke_from_gate9i/distributed_underactivation_by_branch.csv`
- `runs/gate9j_distributed_underactivation_smoke_from_gate9i/distributed_underactivation_status.json`
- `runs/gate9j_distributed_underactivation_smoke_from_gate9i/gate9j_distributed_underactivation_read.md`

## 1. Source And Bind

This smoke run consumes the first Gate9I dominance bundle:

- `source_gate9i_run_id = gate9i_support_anchor_cleaner_cell_dominance_smoke_from_gate9h`
- `source_gate9i_code_git_commit = 29ab3cdcfb4f0e182f85515f0360755b636ad6a8`

The Gate9J audit bind is:

- `method_id = gate9j_distributed_underactivation_audit_v1`
- `code_git_commit = 76e34d4edb9b6278270e08cf5f509a25c155724c`

The audited redesign candidate remains:

- `anchor_coverage_gap_abs_v1`

## 2. What Landed

Gate9J does not invent a new object.

It holds the Gate9H redesign candidate fixed and decomposes the conflict-side split named by Gate9I.

The question is only:

- where does support-anchor `distributed_underactivation` actually live

The audit therefore compares:

- direct versus distributed support-side means
- consistent versus nonconsistent answer branches inside the distributed cell
- answer-coverage and token-coverage deltas against the direct consistent baseline

## 3. Smoke Read

### 3.1 Underactivation Still Holds At Cell Level

The support-side means are:

- `direct_mean_gap = 0.136027`
- `distributed_mean_gap = 0.089808`

That keeps:

- `distributed_underactivation_status = triggered`

So Gate9I's named subblocker is still real after direct audit.

### 3.2 The Distributed Cell Splits By Branch

The decisive split is:

- `distributed_consistent_gap = 0.055232`
- `distributed_nonconsistent_gap = 0.124384`

That yields:

- `distributed_answer_target_split_status = triggered`
- `distributed_consistent_branch_status = underactivated`

So the blocker is not honestly described as family-wide weakness across all distributed rows.

It concentrates on the distributed consistent-answer branch.

### 3.3 Answer-Side Suppression Leads The Gap Loss

Against the direct consistent baseline:

- `direct_to_distributed_consistent_gap_delta = -0.079266`
- `direct_to_distributed_consistent_answer_delta = -0.099635`
- `direct_to_distributed_consistent_token_delta = -0.020368`

This closes as:

- `direct_baseline_answer_suppression_status = triggered`
- `gap_loss_explained_as_token_only_status = denied`

So the present compression pattern is better read as answer-side suppression than as token-only carryover.

### 3.4 Current Subblocker

The status payload closes as:

- `next_named_subblocker = distributed_consistent_answer_compression`

This is the right Gate9J landing point.

Gate9J does not solve distributed underactivation.

It names the narrower branch where that underactivation now lives.

## 4. Current Scientific Judgment

The correct Gate9J smoke judgment is:

- Gate9J succeeded as a subblocker-decomposition audit
- support-anchor distributed underactivation remains real
- it is not family-wide across the distributed cell
- it concentrates on the distributed consistent-answer branch
- the present gap loss is led by answer-side suppression rather than token-only carryover

The strongest honest sentence is:

- `Gate9J shows that support-anchor distributed underactivation is concentrated in the distributed consistent-answer branch, with answer-side suppression dominating the observed gap loss.`

## 5. What This Smoke Run Earns

This smoke run earns the right to say:

- the support-anchor conflict-side split is now internally decomposed one layer further
- future work can attack `distributed_consistent_answer_compression` directly as the next named subblocker
- the line no longer needs to treat distributed underactivation as a vague family-wide phenomenon

## 6. What This Smoke Run Does Not Earn

This smoke run does not earn the right to say:

- the support-anchor redesign line is now healthy
- cleaner-cell dominance is resolved
- operator admission should open
- answer coverage has already been repaired

## 7. Next Honest Move

The next honest move is not:

- another abstract blocker note
- operator opening
- a new metric family

The next honest move is:

- attack `distributed_consistent_answer_compression` as the next named subblocker inside the support-anchor line
