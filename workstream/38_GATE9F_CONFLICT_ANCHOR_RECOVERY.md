# Gate9F Conflict-Anchor Recovery

Status: narrow actual-recovery spec, first implementation landed on the existing lane
Role: Gate9F recovery spec, not operator reopening or geometry redesign
Date: 2026-03-21

Gate9F proceeds from:

- `36_GATE9E_CONFLICT_ANCHOR_MATERIALIZATION.md`
- `37_GATE9E_CONFLICT_ANCHOR_MATERIALIZATION_SMOKE.md`

The first Gate9F recovery consumer now exists in:

- `tools/run_gate9f_conflict_anchor_recovery.py`

The first tracked Gate9F smoke execution read is now recorded in:

- `39_GATE9F_CONFLICT_ANCHOR_RECOVERY_SMOKE.md`

## 0. Why This Exists

Gate9E already earned the dry-run sentence.

What is now known is:

- law change is not required
- geometry change is not required
- answer-target split does not fork the target
- cleaner-side spill is not required
- the target text source is `world_truth.distributed_block_claim`

So the next honest move is no longer:

- another admission note
- another abstract coverage debate
- another geometry story

It is:

- actual recovery of the missing `conflict_anchor` branch on the existing lane

## 1. Scope

Gate9F covers only:

- `distributed_incompatibility`
- the existing `conflict_anchor` artifact lane
- actual materialization of the missing conflict-anchor branch
- post-recovery rerun of Gate9D and Gate9C only

It does not:

- reopen Gate8 court
- add a new cycle motif
- add a new closure convention
- redesign anchors
- open any graph-wide operator

## 2. Public Question

The Gate9F question is:

- can the declared conflict chunk be materialized into the existing conflict-anchor artifact lane and then recover coverage under the frozen law

More concretely:

- write the missing `conflict_anchor.txt`
- write the matching `conflict_anchor_meta.json`
- write the matching `conflict_anchor_triplets.ndjson`
- rerun the narrow downstream slices that judge coverage and admission

## 3. Recovery Discipline

The recovery scope stays extremely narrow.

Only these rows are in scope:

- `distributed_incompatibility` rows from the Gate9E candidate registry

The target text source stays fixed:

- `world_truth.distributed_block_claim`

The recovery lane stays fixed:

- `conflict_anchor.txt`
- `conflict_anchor_meta.json`
- `conflict_anchor_triplets.ndjson`

The recovery contract is:

- no new extraction semantics
- no new closure rule
- no new cycle family
- no cleaner-side edits

## 4. Post-Recovery Judgment

After materialization, Gate9F judges only:

- Gate9D `coverage_recovery_status`
- Gate9C `usable_motif_coverage_status`
- Gate9C `operator_admission_status`

Gate9A and Gate9B may be rerun as prerequisites.

They are not reopened as public verdict layers.

## 5. Gate9F Falsifiers

Gate9F must keep these falsifiers explicit:

- materialization does not recover `conflict_answer_terminal_token_cycle` coverage
- recovery spills into cleaner-side semantics
- declaration itself turns out unstable under actual materialization
- actual recovery requires a new closure convention after all
- the blocker turns out not to live in the existing lane

## 6. What Gate9F Can Earn

At most, Gate9F can earn the right to say:

- the missing conflict-anchor branch has been materialized on the existing lane
- Gate9D coverage either does or does not recover under the frozen law
- Gate9C usable motif coverage either does or does not recover under the frozen law
- operator admission may still remain closed even after successful coverage recovery

It still does not earn:

- Gate8 court rerun
- operator opening
- new geometry language
- anchor redesign

## 7. Current Memory Hook

The shortest acceptable sentence is:

- recover the missing branch first, then judge only coverage and admission
