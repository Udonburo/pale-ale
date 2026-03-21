# Gate9D Conflict Motif Coverage

Status: narrow coverage-recovery spec, first implementation landed
Role: Gate9D coverage-recovery spec, not operator opening or anchor redesign
Date: 2026-03-21

Gate9D proceeds from:

- `31_GATE9C_OPERATOR_ADMISSION.md`
- `32_GATE9C_MISSINGNESS_TOPOLOGY.md`
- `33_GATE9C_MISSINGNESS_AUDIT_SMOKE.md`

Initial Gate9D coverage-recovery audit implementation now exists in:

- `tools/run_gate9d_conflict_motif_coverage_audit.py`

The first tracked Gate9D smoke execution read is now recorded in:

- `35_GATE9D_CONFLICT_MOTIF_COVERAGE_SMOKE.md`

## 0. Why This Exists

Gate9C named the blocker.

The current denied admission object is not abstract anymore.

It is:

- `distributed_incompatibility / conflict_answer_terminal_token_cycle`

The next honest move is therefore not:

- operator design
- anchor-conditioned redesign
- larger graph machinery

It is:

- auditing whether the named bundle-specific coverage gap can be recovered under the frozen law

## 1. Scope

Gate9D studies only:

- `conflict_answer_terminal_token_cycle`
- conflict-side coverage gaps
- recovery-candidate status under the frozen law

It does not:

- open any graph-wide operator
- redesign anchor-conditioned closure
- change node or edge ontology
- add new cycle motifs
- change the frozen graph-gauge law

## 2. Public Question

The Gate9D question is:

- can the current bundle-specific conflict-cycle gap be recovered without changing the law

More concretely:

- where exactly does conflict-cycle coverage fail upstream
- is that failure recoverable inside the current law
- would any attempted recovery contaminate cleaner-side semantics

## 3. Focus Object

The only focus motif is:

- `conflict_answer_terminal_token_cycle`

The only target gap Gate9D is allowed to pursue is:

- missing conflict-cycle availability on `distributed_incompatibility`

If other weaknesses appear, they may be recorded.

They may not expand Gate9D scope.

## 4. Public Coverage Registry

Gate9D must emit a deterministic registry for every row of the focus motif.

Each row must include at least:

- `cell_id`
- `answer_target_type`
- `cycle_outcome`
- `absence_class`
- whether conflict chunks are declared upstream
- whether conflict-anchor artifacts are materialized
- whether answer-triplet artifacts are materialized
- a deterministic `recovery_path_status`

The public point is not a new scalar.

It is an object-level statement of:

- what exists
- what is missing
- what kind of recovery, if any, is licensed

## 5. Recovery Path Statuses

Every focus-motif row must land in exactly one recovery-path status.

The required public statuses are:

- `already_covered`
- `not_applicable_structural`
- `recoverable_under_frozen_law_candidate`
- `blocked_without_law_change`
- `implementation_bound_gap`

An additional taxonomic status may appear if the bundle demands it, but these five are mandatory.

### 5.1 Already Covered

Use `already_covered` when the conflict-cycle row is already available on the active bundle.

### 5.2 Not Applicable Structural

Use `not_applicable_structural` when the focus motif is absent on cleaner cells because the frozen law does not license a conflict cycle there.

This keeps cleaner-side absence from being mistaken for a recoverable failure.

### 5.3 Recoverable Under Frozen Law Candidate

Use `recoverable_under_frozen_law_candidate` when:

- the row is conflict-intended
- conflict material is declared upstream
- the focus motif is still missing
- the gap can be named as missing bundle materialization rather than missing legal structure

This is the main positive Gate9D status.

It names a candidate for honest recovery.

It does not yet count as recovery itself.

### 5.4 Blocked Without Law Change

Use `blocked_without_law_change` when the focus motif is missing and the upstream declaration does not carry the conflict-side object needed to instantiate the motif.

If this status appears on the named blocker, frozen-law recovery is denied.

### 5.5 Implementation Bound Gap

Use `implementation_bound_gap` when the focus motif failure is caused by execution or registry mechanics rather than by bundle semantics.

The canonical cases remain:

- `missing_cycle_edge`
- `missing_terminal_token`

## 6. Gate9D Falsifiers

Gate9D must keep its falsifiers explicit.

The required falsifiers are:

- coverage is still not recovered on the named blocker
- any recovery candidate would pollute cleaner-side semantics
- the gap is actually implementation-bound rather than bundle-specific
- recovery would require an implicit law change

Gate9D succeeds only if these remain readable as explicit statuses rather than being hidden inside one summary score.

## 7. What This Audit Can Earn

At most, Gate9D can earn the right to say:

- the named conflict-cycle blocker is now traced to a specific upstream materialization gap
- frozen-law recovery is either plausibly available or honestly denied
- cleaner-side contamination risk is explicit rather than guessed

It still does not earn:

- actual coverage recovery
- operator admission
- anchor redesign
- graph-wide machinery

## 8. Current Memory Hook

The shortest acceptable sentence is:

- recover coverage honestly before reopening the operator question
