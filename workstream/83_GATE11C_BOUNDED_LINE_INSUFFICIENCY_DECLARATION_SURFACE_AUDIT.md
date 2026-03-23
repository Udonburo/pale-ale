# Gate11C Bounded-Line Insufficiency Declaration-Surface Audit

Status: first implementation landed and first smoke execution recorded
Role: bounded-line insufficiency declaration-surface audit, not candidate declaration, reopening-eligibility judgment, or operator reopening
Date: 2026-03-23

Gate11C proceeds from:

- `76_GATE10_CLOSEOUT.md`
- `78_GATE11_OPERATOR_REOPENING_ELIGIBILITY.md`
- `79_GATE11A_NAMED_OPERATOR_PRESSURE_ADMISSIBILITY.md`
- `80_GATE11A_NAMED_OPERATOR_PRESSURE_ADMISSIBILITY_SMOKE.md`
- `81_GATE11B_BOUNDED_LINE_INSUFFICIENCY_DECLARABILITY.md`
- `82_GATE11B_BOUNDED_LINE_INSUFFICIENCY_DECLARABILITY_SMOKE.md`

The frozen Gate10 closeout-support line remains recorded in:

- `74_GATE10F_PRE_CLOSEOUT_JUDGMENT.md`
- `75_GATE10F_PRE_CLOSEOUT_JUDGMENT_SMOKE.md`

The first Gate11C bounded-line insufficiency declaration-surface audit consumer now exists in:

- `tools/run_gate11c_bounded_line_insufficiency_declaration_surface_audit.py`

The first tracked Gate11C bounded-line insufficiency declaration-surface smoke read is now recorded in:

- `84_GATE11C_BOUNDED_LINE_INSUFFICIENCY_DECLARATION_SURFACE_AUDIT_SMOKE.md`

## 0. Scope

Gate11C is the third narrow Gate11 slice.

Gate11C does:

- ask what would count as a valid explicit declaration surface for one bounded-line insufficiency candidate under the frozen post-Gate10 line
- preserve the Gate10 closeout sentence exactly as already earned
- preserve the Gate11A absence result exactly as already recorded
- preserve the Gate11B no-candidate state exactly as already recorded
- define the narrow conditions that a future explicit declaration would have to satisfy before any candidate may be treated as honestly declared

Gate11C does not:

- declare a bounded-line insufficiency candidate
- decide reopening eligibility
- reopen operator admission
- declare graph-wide operator behavior earned
- settle broader trusted-tree settlement
- retroactively reinterpret Gate9, Gate10, Gate11A, or Gate11B memory
- mine the repo outside the controlling source run
- choose one candidate out of multiple undeclared or multiply-declared possibilities

## 1. Controlling Source Run

Gate11C consumes exactly this controlling source run:

- `runs/gate11b_bounded_line_insufficiency_declarability_smoke_from_gate11a`

No additional source run is in scope.

Under the currently frozen Gate11B source, the recorded upstream result is:

- `bounded_line_insufficiency_candidate_status = absent`
- `bounded_line_insufficiency_class_status = none`
- `bounded_line_insufficiency_declarability_status = not_yet_declarable`

So Gate11C must treat the current controlling source as a no-candidate preservation source.

Gate11C may define what a future explicit declaration surface would have to look like, but it may not:

- infer that such a declaration already exists
- fabricate a candidate from earlier Gate10 or Gate9 material
- resolve ambiguity by worker-side completion

## 2. Public Question

The Gate11C question is:

- `what would count as a valid explicit declaration surface for one bounded-line insufficiency candidate under the frozen post-Gate10 line?`

This is narrower than:

- candidate declaration itself
- reopening-eligibility judgment
- operator reopening
- graph-wide operator retry

It is only:

- the declaration-surface gate for what would count as one valid explicit bounded-line insufficiency declaration later
- and, under the current frozen source, the audit that prevents worker-side invention before such a surface exists

## 3. Why Gate11C Exists

Gate11A earned:

- the post-Gate10 line has now been checked for a real named operator-pressure case
- the correct current result is `absent / none / not_yet_admissible`

Gate11B earned:

- the frozen Gate11A source has now been checked for one explicitly declarable bounded-line insufficiency candidate
- the correct current result is `absent / none / not_yet_declarable`

So the next honest move is not:

- invent a candidate
- declare reopening eligible
- reopen the operator line

It is:

- define the narrow declaration surface that a future candidate declaration would have to satisfy
- so that a later slice can judge whether one explicit declaration exists without worker-side invention

## 4. Declaration-Surface Discipline

Gate11C must define surface before candidate.

A future bounded-line insufficiency declaration counts as valid only if all of the following are fixed in advance:

- an explicit marker shape is defined
- single-candidate singularity is defined
- bounded-line insufficiency evidence shape is defined
- anti-inflation boundary is defined

If any of those conditions remains undefined, Gate11C must not treat a future narrative mention as a valid declaration surface.

## 5. Valid Declaration Surface Conditions

### 5.1 Explicit Marker Shape

A future candidate counts as explicitly declared only if the same later frozen run contains all of the following matching surfaces:

- one machine-readable status marker `bounded_line_insufficiency_candidate_declaration_status = declared`
- one non-empty stable `bounded_line_insufficiency_candidate_id`
- one `bounded_line_insufficiency_class_status` value from the Gate11B class set
- one registry row keyed to that same `bounded_line_insufficiency_candidate_id`
- one read sentence or bullet using the fixed declaration prefix `one bounded-line insufficiency candidate is explicitly declared:`

The same `bounded_line_insufficiency_candidate_id` must match across status payload, registry row, and read sentence.

The following do not count as declaration surface:

- narrative mention without a declaration marker
- suggestive prose such as `maybe`, `could be`, or `seems like`
- class naming without a declared candidate id
- worker-side reconstruction across multiple files

### 5.2 Single-Candidate Singularity

A valid declaration surface may carry only one candidate in one run.

So a future declaration counts only if:

- exactly one `bounded_line_insufficiency_candidate_id` is declared in the run
- exactly one Gate11B class is attached to that declared candidate
- status payload, registry row, and read sentence all point to the same single candidate

The following must be treated as `deferred` rather than declared:

- multiple candidate ids in one run
- one candidate id paired with multiple unresolved classes
- inconsistent candidate identity across output surfaces

Gate11C does not allow:

- worker-side candidate selection
- class disambiguation by interpretive completion

### 5.3 Bounded-Line Insufficiency Evidence Shape

A valid declaration surface must do more than express dissatisfaction.

A future declaration counts only if the same later frozen run states explicitly that:

- the current bounded line cannot honestly host the named candidate
- the insufficiency is about the bounded line's expressive or structural limit
- the insufficiency is attached to the declared candidate id rather than to general discomfort

The minimum honest evidence surface is:

- one machine-readable status marker `bounded_line_host_failure_status = explicit`
- one read sentence or bullet that explicitly states `the current bounded line cannot honestly host <candidate_id>`

The following do not count:

- generic dissatisfaction
- desire for stronger language
- desire for cleaner theory
- pressure inferred only from repeated failed remedies

### 5.4 Anti-Inflation Boundary

A valid declaration surface must remain narrow and non-promotional.

A future declaration counts only if the same later frozen run keeps all of the following explicit:

- `broader_trusted_tree_settlement_still_unearned_status = confirmed`
- `retroactive_reinterpretation_forbidden_status = confirmed`
- `settlement_inflation_pressure_status = absent`
- `graph_wide_operator_leap_pressure_status = absent`

So the declaration surface must not depend on:

- broader trusted-tree settlement promotion
- retroactive rewrite
- graph-wide leap

If the declaration surface becomes legible only by relying on one of those moves, Gate11C must treat it as:

- `denied`

## 6. Required Judgment Checks

Gate11C may define a valid declaration surface only if all of the following remain clear.

### 6.1 Gate10 Closeout Preservation

The declaration-surface audit is invalid if:

- the Gate10 closeout sentence no longer remains preserved as recorded

### 6.2 Gate11A Absence Result Preservation

The declaration-surface audit is invalid if:

- the Gate11A absence result no longer remains preserved as recorded

### 6.3 Gate11B No-Candidate State Preservation

The declaration-surface audit is invalid if:

- the Gate11B no-candidate state no longer remains preserved as recorded

### 6.4 Explicit Marker Shape Is Defined

The declaration-surface audit is invalid if:

- no narrow marker shape is fixed for what would count as explicit declaration

### 6.5 Single-Candidate Singularity Is Defined

The declaration-surface audit is invalid if:

- the spec would allow multiple candidate ids or unresolved class ambiguity inside one run

### 6.6 Bounded-Line Insufficiency Evidence Shape Is Defined

The declaration-surface audit is invalid if:

- a future declaration could pass without explicitly stating that the current bounded line cannot honestly host the candidate

### 6.7 Operator Admission Still Denied

The declaration-surface audit is invalid if:

- the surface definition assumes operator reopening instead of preserving operator admission as still denied

### 6.8 Anti-Inflation Boundary Is Defined

The declaration-surface audit is invalid if:

- a future declaration could pass while depending on broader-settlement promotion, retroactive rewrite, or graph-wide leap pressure

## 7. Falsifiers

Gate11C fails declaration-surface definition if any of the following happens:

- Gate10 closeout preservation fails
- Gate11A absence-result preservation fails
- Gate11B no-candidate preservation fails
- explicit declaration marker shape is left narrative-only or inferential
- single-candidate singularity is left ambiguous
- bounded-line insufficiency evidence is allowed to collapse into generic dissatisfaction
- operator reopening is assumed rather than kept denied
- declaration surface depends on settlement inflation, retroactive rewrite, or graph-wide leap pressure
- repo-wide mining or worker-side synthesis is required to make the declaration legible

## 8. Expected Outputs

Any Gate11C implementation must emit exactly these files:

- `manifest.json`
- `checksums.json`
- `bounded_line_insufficiency_declaration_surface_registry.jsonl`
- `bounded_line_insufficiency_declaration_surface_policy_compare.csv`
- `bounded_line_insufficiency_declaration_surface_status.json`
- `gate11c_bounded_line_insufficiency_declaration_surface_read.md`

## 9. Required Status Keys

Any Gate11C implementation must emit explicit status for:

- `gate10_closeout_preservation_status`
- `gate11a_absence_result_preservation_status`
- `gate11b_no_candidate_state_preservation_status`
- `broader_trusted_tree_settlement_still_unearned_status`
- `operator_admission_still_denied_status`
- `retroactive_reinterpretation_forbidden_status`
- `explicit_marker_shape_status`
- `single_candidate_singularity_status`
- `bounded_line_insufficiency_evidence_shape_status`
- `anti_inflation_boundary_status`
- `bounded_line_insufficiency_declaration_surface_status`
- `next_named_blocker`

## 10. Status Space

Gate11C is limited to the following judgment space.

### 10.1 Preservation Statuses

Each of:

- `gate10_closeout_preservation_status`
- `gate11a_absence_result_preservation_status`
- `gate11b_no_candidate_state_preservation_status`

must be emitted as one of:

- `preserved`
- `not_preserved`

### 10.2 Boundary Statuses

Each of:

- `broader_trusted_tree_settlement_still_unearned_status`
- `operator_admission_still_denied_status`
- `retroactive_reinterpretation_forbidden_status`

must be emitted as one of:

- `confirmed`
- `not_confirmed`

### 10.3 Definition Sub-Statuses

Each of:

- `explicit_marker_shape_status`
- `single_candidate_singularity_status`
- `bounded_line_insufficiency_evidence_shape_status`
- `anti_inflation_boundary_status`

must be emitted as one of:

- `defined`
- `not_yet_defined`
- `denied`
- `deferred`

### 10.4 Declaration-Surface Outcome Status

`bounded_line_insufficiency_declaration_surface_status` must be emitted as one of:

- `surface_defined`
- `not_yet_defined`
- `denied`
- `deferred`

`surface_defined` means only:

- the future declaration surface for one bounded-line insufficiency candidate has been fixed narrowly enough to audit later

It does not mean:

- a candidate is already declared
- reopening is eligible
- operator reopening is earned

## 11. Outcome Ladder

Gate11C outcomes are limited to these four.

### 11.1 Surface Defined

Use `surface_defined` only if:

- explicit marker shape is defined
- single-candidate singularity is defined
- bounded-line insufficiency evidence shape is defined
- anti-inflation boundary is defined
- no falsifier fires

### 11.2 Not Yet Defined

Use `not_yet_defined` if:

- the frozen post-Gate10 line still preserves absence
- but the declaration surface has not yet been fixed narrowly enough to audit later candidate declaration honestly

### 11.3 Denied

Use `denied` if:

- the proposed declaration surface works only by allowing inflation, rewrite, graph-wide leap, or worker-side invention

### 11.4 Deferred

Use `deferred` if:

- the frozen source evidence is incomplete or contradictory in a way that blocks declaration-surface definition
- or the source would require worker-side resolution of multiple possible declaration surfaces

## 12. Forbidden

The following remain forbidden in Gate11C:

- no candidate declaration itself
- no reopening-eligibility judgment
- no operator reopening
- no graph-wide operator
- no broader trusted-tree settlement promotion
- no retroactive rewrite of Gate9, Gate10, Gate11A, or Gate11B
- no repo-wide mining outside the controlling source run
- no worker-side inference in place of explicit declaration markers
- no multiple-candidate resolution
- no scalar comeback
- no benchmark-zoo expansion
- no sheaf branding
- no higher-gauge branding
- no KAGAMI rhetoric inside the public verdict

## 13. Delegation Boundary

An implementation worker may do:

- narrow consumer implementation
- unit tests
- status payload emission
- tracked smoke handoff formatting

An implementation worker may not do:

- candidate declaration
- reopening-eligibility judgment
- operator reopening judgment
- doctrine redesign
- blocker naming redesign
- falsifier redesign
- scope widening
- theory branding

If the spec is insufficient, the work must stop and report the gap rather than invent behavior.

## 14. Memory Hook

The Gate11C sentence is:

- Gate11C does not search for a bounded-line insufficiency candidate
- it defines what would count as a valid explicit declaration surface for one such candidate under the frozen post-Gate10 line

The shortest acceptable memory hook is:

- `Gate11C does not declare a candidate; it fixes the narrow surface that would have to exist before one bounded-line insufficiency candidate could later count as explicitly declared.`
