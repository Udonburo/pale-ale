# Gate9G Anchor-Conditioned Triviality

Status: narrow admission-blocker spec, first implementation planned
Role: Gate9G anchor-conditioned-read blocker spec, not anchor redesign or operator opening
Date: 2026-03-21

Gate9G proceeds from:

- `31_GATE9C_OPERATOR_ADMISSION.md`
- `39_GATE9F_CONFLICT_ANCHOR_RECOVERY_SMOKE.md`

## 0. Why This Exists

Gate9F removed the named coverage blocker.

After recovery, the admission line now reads:

- usable motif coverage = provisionally clear
- missingness topology = clear
- operator admission = still denied

That means the next honest blocker is no longer coverage.

It is:

- the current anchor-conditioned read still has not earned non-trivial status

## 1. Scope

Gate9G studies only:

- the current `anchor_conditioned_closure` observable
- whether its surviving rows are non-trivial or collapsed
- the recovered Gate9A bundle after Gate9F

It does not:

- redesign anchors
- redesign closure
- reopen operator admission
- reopen cleaner-cell dominance

## 2. Public Question

The Gate9G question is:

- is the current anchor-conditioned read failing because the bundle is weak, or because the observable collapses to a trivial object under the present construction

More concretely:

- do surviving rows carry non-zero closure defect
- do conditioned ranks simply saturate the anchor span
- is the current read trivial on the rows where it is actually defined

## 3. Public Object

Gate9G must emit a deterministic row registry that includes at least:

- `anchor_kind`
- `cell_id`
- `closure_outcome`
- `anchor_rank`
- `answer_conditioned_rank`
- `token_conditioned_rank`
- `anchor_answer_coverage`
- `anchor_token_coverage`
- `anchor_conditioned_closure_defect`
- triviality classification

The public object is not a rescue.

It is:

- a blocker audit on whether the current anchor-conditioned read is mathematically trivial on the active bundle

## 4. Gate9G Falsifiers

Gate9G must keep these falsifiers explicit:

- non-missing rows carry only zero-level closure defect
- those rows simply saturate the anchor span
- no non-trivial candidate rows remain after recovery

If those remain true, then `non-trivial anchor-conditioned read` stays denied.

## 5. What This Audit Can Earn

At most, Gate9G can earn the right to say:

- the current anchor-conditioned read is genuinely non-trivial somewhere

Or:

- the current anchor-conditioned read collapses and is itself the next named blocker

It still does not earn:

- anchor redesign
- operator opening
- field language

## 6. Current Memory Hook

The shortest acceptable sentence is:

- name whether the anchor-conditioned read is weak or trivial before trying to fix it
