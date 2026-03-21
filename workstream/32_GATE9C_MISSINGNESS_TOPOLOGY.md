# Gate9C Missingness Topology

Status: narrow admission-audit spec, first implementation landed
Role: Gate9C missingness-topology spec, not operator design or execution verdict
Date: 2026-03-21

Gate9C proceeds from:

- `31_GATE9C_OPERATOR_ADMISSION.md`

Initial Gate9C admission-audit implementation now exists in:

- `tools/run_gate9c_missingness_topology_audit.py`

The first tracked smoke execution read is now recorded in:

- `33_GATE9C_MISSINGNESS_AUDIT_SMOKE.md`

The next narrow coverage-recovery slice is now tracked in:

- `34_GATE9D_CONFLICT_MOTIF_COVERAGE.md`

## 0. Why This Exists

Gate9C begins with admission criteria, not operator design.

The first unmet admission object most directly exposed by Gate9B is:

- missingness topology

The immediate task is therefore not:

- smoothing
- propagation
- graph-wide operator design

It is:

- making absence and motif coverage explicit enough that operator admission can be judged honestly

## 1. Scope

This workstream studies only:

- missingness
- motif coverage
- absence classification

It does not yet redesign:

- anchor-conditioned closure
- cycle family itself
- graph-wide operators

## 2. Public Question

The Gate9C missingness-topology question is:

- where exactly are the current motifs unavailable, and what kind of absence is that

The public object is:

- deterministic missingness and coverage artifacts over the currently licensed cycle family

## 3. Required Missing Outcomes

The first audit must treat the following as first-class missingness states:

- `missing_support_anchor`
- `missing_conflict_anchor`
- `missing_cycle_edge`
- `missing_terminal_token`

Additional non-success outcomes may appear, but these are the mandatory public states.

## 4. Absence Classes

Every missingness row must be classified into exactly one of these classes:

- `structural`
- `taxonomic`
- `bundle_specific`
- `implementation_bound`

The first-pass deterministic reading rules are:

### 4.1 Structural

Use `structural` when the motif is absent because the current cell semantics do not license that motif under the frozen law.

The canonical first case is:

- conflict-cycle absence on cleaner cells that are not conflict-intended

### 4.2 Taxonomic

Use `taxonomic` when availability differs by `answer_target_type` inside the same cell and motif family.

The signature is:

- one answer-target branch carries the motif
- another answer-target branch in the same cell / motif does not

### 4.3 Bundle-Specific

Use `bundle_specific` when the motif is not structurally forbidden, but the current bundle still fails to instantiate it.

This means:

- the motif is conceptually allowed on that cell family
- but the present evidence package does not materialize it

### 4.4 Implementation-Bound

Use `implementation_bound` when the missingness arises from execution or registry mechanics rather than from cell semantics.

The first canonical cases are:

- `missing_cycle_edge`
- `missing_terminal_token`

## 5. Coverage Objects

The audit must emit deterministic coverage artifacts at two levels.

### 5.1 Cell / Motif / Answer-Target Coverage

For each:

- `cell_id`
- `cycle_type`
- `answer_target_type`

the audit must emit:

- `n_rows`
- `n_available`
- `n_missing`
- `coverage_rate`
- dominant missing outcome, if any
- absence-class counts

### 5.2 Cell / Motif Coverage

For each:

- `cell_id`
- `cycle_type`

the audit must emit:

- total availability
- total missingness
- `coverage_rate`
- whether that motif is currently usable on that cell under the present bundle

## 6. Usable Motif Coverage

The first public usable-coverage object is modest.

It is not:

- a global score

It is:

- the fraction of rows on a given cell / motif that are actually available under the present bundle

The current admission burden is asymmetric.

What matters most is:

- conflict-side motif coverage
- especially coverage on `distributed_incompatibility`

## 7. What This Audit Can Earn

At most, this audit can earn the right to say:

- missingness topology is now explicit as an admission object
- usable motif coverage can now be read rather than guessed

It still does not earn:

- operator admission
- holonomy rescue
- closure redesign

## 8. Current Memory Hook

The shortest acceptable sentence is:

- account for absence before attempting propagation
