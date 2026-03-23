# Gate11D One Bounded-Line Insufficiency Explicit-Declaration Audit

Status: first implementation landed and first smoke execution recorded
Role: one bounded-line insufficiency explicit-declaration audit, not reopening-eligibility judgment or operator reopening
Date: 2026-03-23

Gate11D proceeds from:

- `76_GATE10_CLOSEOUT.md`
- `78_GATE11_OPERATOR_REOPENING_ELIGIBILITY.md`
- `79_GATE11A_NAMED_OPERATOR_PRESSURE_ADMISSIBILITY.md`
- `80_GATE11A_NAMED_OPERATOR_PRESSURE_ADMISSIBILITY_SMOKE.md`
- `81_GATE11B_BOUNDED_LINE_INSUFFICIENCY_DECLARABILITY.md`
- `82_GATE11B_BOUNDED_LINE_INSUFFICIENCY_DECLARABILITY_SMOKE.md`
- `83_GATE11C_BOUNDED_LINE_INSUFFICIENCY_DECLARATION_SURFACE_AUDIT.md`
- `84_GATE11C_BOUNDED_LINE_INSUFFICIENCY_DECLARATION_SURFACE_AUDIT_SMOKE.md`

The frozen Gate10 closeout-support line remains recorded in:

- `74_GATE10F_PRE_CLOSEOUT_JUDGMENT.md`
- `75_GATE10F_PRE_CLOSEOUT_JUDGMENT_SMOKE.md`

The first Gate11D one bounded-line insufficiency explicit-declaration audit consumer now exists in:

- `tools/run_gate11d_one_bounded_line_insufficiency_explicit_declaration_audit.py`

The first tracked Gate11D one bounded-line insufficiency explicit-declaration smoke read is now recorded in:

- `86_GATE11D_ONE_BOUNDED_LINE_INSUFFICIENCY_EXPLICIT_DECLARATION_AUDIT_SMOKE.md`

## 0. Scope

Gate11D is the fourth narrow Gate11 slice.

Gate11D does:

- ask whether one explicitly declared bounded-line insufficiency candidate now exists under the fixed Gate11C declaration surface
- preserve the Gate10 closeout sentence exactly as already earned
- preserve the Gate11A absence result exactly as already recorded
- preserve the Gate11C `surface_defined` result exactly as already recorded unless the same frozen source explicitly instantiates one declaration
- decide only whether one explicit bounded-line insufficiency declaration is actually present

Gate11D does not:

- define the declaration surface again
- decide reopening eligibility
- reopen operator admission
- declare graph-wide operator behavior earned
- settle broader trusted-tree settlement
- retroactively reinterpret Gate9, Gate10, Gate11A, Gate11B, or Gate11C memory
- mine the repo outside the controlling source run
- choose one candidate out of multiple explicit or partially explicit possibilities

## 1. Controlling Source Run

Gate11D consumes exactly this controlling source run:

- `runs/gate11c_bounded_line_insufficiency_declaration_surface_audit_smoke_from_gate11b`

No additional source run is in scope.

Under the currently frozen Gate11C source, the recorded upstream result is:

- `explicit_marker_shape_status = defined`
- `single_candidate_singularity_status = defined`
- `bounded_line_insufficiency_evidence_shape_status = defined`
- `anti_inflation_boundary_status = defined`
- `bounded_line_insufficiency_declaration_surface_status = surface_defined`

So Gate11D must treat the current controlling source as:

- a surface-defined source
- but not yet an explicit candidate-declaration source unless the same frozen run actually instantiates that surface

The worker must not:

- mine earlier Gate10 bundles directly
- mine Gate9 docs directly
- infer a declaration from narrative pressure alone
- select one candidate when the source exposes more than one explicit or partially explicit option

## 2. Public Question

The Gate11D question is:

- `does one explicitly declared bounded-line insufficiency candidate now exist under the fixed Gate11C declaration surface?`

This is narrower than:

- declaration-surface definition
- reopening-eligibility judgment
- operator reopening
- graph-wide operator retry

It is only:

- the explicit-declaration gate for whether one bounded-line insufficiency candidate is actually explicit now
- and, under the current frozen source, the audit that preserves `not_yet_declared` unless that explicit declaration truly exists

## 3. Why Gate11D Exists

Gate11A earned:

- the correct current result is `absent / none / not_yet_admissible`

Gate11B earned:

- the correct current result is `absent / none / not_yet_declarable`

Gate11C earned:

- the future declaration surface is now fixed narrowly enough to audit later
- the correct current result is `surface_defined`
- no candidate is declared there

So the next honest move is not:

- redefine declaration surface
- declare reopening eligible
- reopen the operator line

It is:

- ask whether one candidate declaration actually exists under that already fixed Gate11C surface

## 4. Explicit-Declaration Discipline

Gate11D must audit declaration, not invent it.

One bounded-line insufficiency candidate counts as explicitly declared only if all of the following are true in the same frozen controlling source run:

- the Gate11C explicit declaration marker is actually present
- exactly one candidate id is carried by that declaration
- exactly one class is carried by that declaration
- the source explicitly states `the current bounded line cannot honestly host <candidate_id>`

If any of those conditions is absent, Gate11D must not promote the source into `declared`.

Narrative mention may not be upgraded into declaration by:

- worker-side inference
- class completion
- repo-wide mining
- rhetorical completion

## 5. Required Explicit Declaration Conditions

### 5.1 Explicit Declaration Marker Is Present

Gate11D does not redefine marker shape.

It asks whether the Gate11C-defined surface is actually instantiated in the controlling source run.

So one declaration counts only if the source run contains all of the following explicit declaration markers:

- `bounded_line_insufficiency_candidate_declaration_status = declared`
- one non-empty `bounded_line_insufficiency_candidate_id`
- one registry row keyed to that same `bounded_line_insufficiency_candidate_id`
- one read sentence or bullet using the prefix `one bounded-line insufficiency candidate is explicitly declared:`

Narrative mention without those markers does not count.

### 5.2 Candidate Id Is Single

One declaration counts only if:

- exactly one `bounded_line_insufficiency_candidate_id` is instantiated in the controlling source run
- status payload, registry row, and read sentence all point to that same single id

The following must be treated as `deferred` rather than `declared`:

- multiple candidate ids
- inconsistent candidate identity across output surfaces

### 5.3 Class Is Single

One declaration counts only if:

- exactly one bounded-line insufficiency class is attached to the declared candidate
- that class is one of the Gate11B class set

The following must be treated as `deferred` rather than `declared`:

- multiple unresolved classes
- class mismatch across output surfaces

### 5.4 Host-Failure Statement Is Explicit

One declaration counts only if the controlling source run explicitly states:

- `the current bounded line cannot honestly host <candidate_id>`

The minimum honest evidence is:

- one machine-readable status marker `bounded_line_host_failure_status = explicit`
- one read sentence or bullet that explicitly states the host-failure sentence for the same candidate id

The following do not count:

- generic dissatisfaction
- desire for stronger language
- pressure inferred only from repeated failed remedies

## 6. Required Judgment Checks

Gate11D may recognize one explicit bounded-line insufficiency declaration only if all of the following remain clear.

### 6.1 Gate10 Closeout Preservation

The explicit-declaration audit is invalid if:

- the Gate10 closeout sentence no longer remains preserved as recorded

### 6.2 Gate11A Absence Result Preservation

The explicit-declaration audit is invalid if:

- the Gate11A absence result no longer remains preserved as recorded

### 6.3 Gate11C Surface Definition Remains Preserved

The explicit-declaration audit is invalid if:

- the Gate11C declaration surface is no longer preserved as `surface_defined`

### 6.4 Broader Settlement Still Unearned

The explicit-declaration audit is invalid if:

- the declaration depends on promoting broader trusted-tree settlement into something already earned

### 6.5 Operator Admission Still Denied

The explicit-declaration audit is invalid if:

- the declaration assumes operator reopening instead of preserving operator admission as still denied

### 6.6 Retroactive Reinterpretation Remains Forbidden

The explicit-declaration audit is invalid if:

- the declaration appears only after rewriting Gate9, Gate10, Gate11A, Gate11B, or Gate11C memory

### 6.7 Explicit Declaration Marker Is Present

The explicit-declaration audit is invalid if:

- the Gate11C-defined explicit declaration marker is not actually instantiated in the controlling source

### 6.8 Candidate Id Is Single

The explicit-declaration audit is invalid if:

- the declaration does not carry exactly one candidate id

### 6.9 Class Is Single

The explicit-declaration audit is invalid if:

- the declaration does not carry exactly one class

### 6.10 Host-Failure Statement Is Explicit

The explicit-declaration audit is invalid if:

- the source does not explicitly state that the current bounded line cannot honestly host the declared candidate

### 6.11 Anti-Inflation Boundary Remains Intact

The explicit-declaration audit is invalid if:

- inflation, rewrite, or graph-wide leap pressure is needed to make the declaration legible

## 7. Falsifiers

Gate11D fails explicit declaration if any of the following happens:

- Gate10 closeout preservation fails
- Gate11A absence-result preservation fails
- Gate11C surface-defined preservation fails
- broader-settlement inflation is needed
- operator reopening is assumed rather than deferred
- retroactive reinterpretation pressure appears
- no Gate11C-defined explicit declaration marker exists in the controlling source
- more than one candidate id is instantiated
- more than one unresolved class is instantiated
- the host-failure statement is absent
- graph-wide operator leap pressure appears
- worker-side resolution would be required

## 8. Expected Outputs

Any Gate11D implementation must emit exactly these files:

- `manifest.json`
- `checksums.json`
- `bounded_line_insufficiency_explicit_declaration_registry.jsonl`
- `bounded_line_insufficiency_explicit_declaration_policy_compare.csv`
- `bounded_line_insufficiency_explicit_declaration_status.json`
- `gate11d_one_bounded_line_insufficiency_explicit_declaration_read.md`

## 9. Required Status Keys

Any Gate11D implementation must emit explicit status for:

- `gate10_closeout_preservation_status`
- `gate11a_absence_result_preservation_status`
- `gate11c_declaration_surface_preservation_status`
- `broader_trusted_tree_settlement_still_unearned_status`
- `operator_admission_still_denied_status`
- `retroactive_reinterpretation_forbidden_status`
- `anti_inflation_boundary_status`
- `bounded_line_insufficiency_explicit_declaration_marker_status`
- `bounded_line_insufficiency_candidate_id_singularity_status`
- `bounded_line_insufficiency_class_singularity_status`
- `bounded_line_host_failure_statement_status`
- `one_bounded_line_insufficiency_explicit_declaration_status`
- `next_named_blocker`

## 10. Status Space

Gate11D is limited to the following judgment space.

### 10.1 Preservation Statuses

Each of:

- `gate10_closeout_preservation_status`
- `gate11a_absence_result_preservation_status`
- `gate11c_declaration_surface_preservation_status`

must be emitted as one of:

- `preserved`
- `not_preserved`

### 10.2 Boundary Statuses

Each of:

- `broader_trusted_tree_settlement_still_unearned_status`
- `operator_admission_still_denied_status`
- `retroactive_reinterpretation_forbidden_status`
- `anti_inflation_boundary_status`

must be emitted as one of:

- `confirmed`
- `not_confirmed`

### 10.3 Explicit Marker Status

`bounded_line_insufficiency_explicit_declaration_marker_status` must be emitted as one of:

- `present`
- `absent`
- `deferred`

### 10.4 Candidate Id Singularity Status

`bounded_line_insufficiency_candidate_id_singularity_status` must be emitted as one of:

- `single`
- `absent`
- `multiple`
- `deferred`

### 10.5 Class Singularity Status

`bounded_line_insufficiency_class_singularity_status` must be emitted as one of:

- `single`
- `none`
- `multiple`
- `deferred`

### 10.6 Host-Failure Statement Status

`bounded_line_host_failure_statement_status` must be emitted as one of:

- `explicit`
- `absent`
- `deferred`

### 10.7 Explicit-Declaration Outcome Status

`one_bounded_line_insufficiency_explicit_declaration_status` must be emitted as one of:

- `declared`
- `not_yet_declared`
- `denied`
- `deferred`

`declared` means only:

- one bounded-line insufficiency candidate is explicitly declared under the fixed Gate11C surface

It does not mean:

- reopening is eligible
- operator reopening is earned

## 11. Outcome Ladder

Gate11D outcomes are limited to these four.

### 11.1 Declared

Use `declared` only if:

- the Gate11C explicit declaration marker is present
- exactly one candidate id is instantiated
- exactly one class is instantiated
- the host-failure statement is explicit
- no falsifier fires

### 11.2 Not Yet Declared

Use `not_yet_declared` if:

- the Gate11C declaration surface remains fixed
- but the controlling source still does not instantiate one full explicit declaration under that surface

This is the expected default result under the current frozen Gate11C source unless a later tracked source supersedes it.

### 11.3 Denied

Use `denied` if:

- the declaration depends on inflation, retroactive rewrite, or graph-wide leap pressure

### 11.4 Deferred

Use `deferred` if:

- the frozen source evidence is incomplete or contradictory in a way that blocks explicit-declaration judgment
- or multiple candidate ids or classes would require worker-side resolution

## 12. Forbidden

The following remain forbidden in Gate11D:

- no declaration-surface redesign
- no reopening-eligibility judgment
- no operator reopening
- no graph-wide operator
- no broader trusted-tree settlement promotion
- no retroactive rewrite of Gate9, Gate10, Gate11A, Gate11B, or Gate11C
- no repo-wide mining outside the controlling source run
- no worker-side inference in place of explicit declaration
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

- declaration-surface redesign
- candidate invention
- reopening-eligibility judgment
- operator reopening judgment
- doctrine redesign
- blocker naming redesign
- falsifier redesign
- scope widening
- theory branding

If the spec is insufficient, the work must stop and report the gap rather than invent behavior.

## 14. Memory Hook

The Gate11D sentence is:

- Gate11D does not define declaration surface again
- it asks whether one bounded-line insufficiency candidate is actually explicit now under the fixed Gate11C surface

The shortest acceptable memory hook is:

- `Gate11D does not redefine declaration surface; it asks whether one bounded-line insufficiency candidate is actually explicit yet under the fixed Gate11C surface.`
