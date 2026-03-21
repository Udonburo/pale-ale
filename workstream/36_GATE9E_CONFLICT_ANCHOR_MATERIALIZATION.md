# Gate9E Conflict-Anchor Materialization

Status: narrow materialization-audit spec, first implementation landed
Role: Gate9E upstream artifact-lane spec, not coverage recovery or operator reopening
Date: 2026-03-21

Gate9E proceeds from:

- `34_GATE9D_CONFLICT_MOTIF_COVERAGE.md`
- `35_GATE9D_CONFLICT_MOTIF_COVERAGE_SMOKE.md`

Initial Gate9E dry-run materialization audit now exists in:

- `tools/run_gate9e_conflict_anchor_materialization_audit.py`

The first tracked Gate9E smoke execution read is now recorded in:

- `37_GATE9E_CONFLICT_ANCHOR_MATERIALIZATION_SMOKE.md`

## 0. Why This Exists

Gate9D already named the blocker and its current signature.

The active blocker is:

- `distributed_incompatibility / conflict_answer_terminal_token_cycle`

The active signature is:

- `declared_conflict_chunk_without_materialized_conflict_anchor`

So the next honest move is not:

- new geometry
- new cycle family
- operator rescue

It is:

- tracing whether the declared conflict chunk can be carried into the existing conflict-anchor artifact lane without changing the law

## 1. Scope

Gate9E studies only:

- `distributed_incompatibility`
- the existing `conflict_anchor` artifact lane
- dry-run materialization candidacy

It does not:

- rewrite the frozen graph-gauge law
- add cycle motifs
- redesign closure
- reopen operator admission
- claim coverage recovery before rerun

## 2. Public Question

The Gate9E question is:

- can the declared conflict chunk be materialized into the existing conflict-anchor artifact lane without changing the law

More concretely:

- which upstream source row carries the missing conflict anchor
- what target text the current lane would need
- which expected files are absent
- whether a deterministic dry-run target can be emitted without polluting cleaner cells

## 3. Focus Object

The only focus object is:

- the conflict-anchor materialization gap on `distributed_incompatibility`

The first implementation may use controls only to explain the lane.

It may not broaden the public judgment beyond the named blocker.

## 4. Public Materialization Registry

Gate9E must emit a deterministic registry for every in-scope row.

Each row must include at least:

- `benchmark_sample_id`
- `answer_target_type`
- declared conflict chunk ids and texts
- the expected conflict-anchor target text
- the source field for that target text
- which conflict-anchor files are actually missing
- whether the declared chunk already contains the expected target text
- whether a dry-run target was emitted

The public object is not recovery itself.

It is:

- an explicit upstream materialization audit

## 5. Dry-Run Materialization

The first implementation is allowed to emit only:

- deterministic dry-run target text
- dry-run target path
- dry-run status

It is not allowed yet to:

- generate new triality triplets
- alter the execution bundle in place
- reopen the candidate execution line

## 6. Gate9E Falsifiers

Gate9E must keep these falsifiers explicit:

- dry-run materialization would still not plausibly restore cycle coverage
- the dry-run spills into cleaner-side semantics
- declaration itself is unstable or missing
- materialization would require a new closure convention
- answer-target branches force different conflict-anchor targets

These must remain public statuses, not narrative footnotes.

## 7. What This Audit Can Earn

At most, Gate9E can earn the right to say:

- the missing conflict anchor is now traced to a specific upstream artifact-lane gap
- a dry-run target can or cannot be emitted under the existing law
- answer-target split either does or does not fork the materialization target

It still does not earn:

- actual coverage recovery
- Gate9D closure
- Gate9C operator admission
- a new geometry line

## 8. Current Memory Hook

The shortest acceptable sentence is:

- fix the missing artifact lane before asking the geometry to explain it
