# Gate11E Explicit-Declaration Instantiation Path Audit

Status: first implementation landed and first smoke execution recorded
Role: explicit-declaration instantiation path audit, not candidate declaration, explicit-declaration audit, reopening-eligibility judgment, or operator reopening
Date: 2026-03-23

Gate11E proceeds from:

- `76_GATE10_CLOSEOUT.md`
- `78_GATE11_OPERATOR_REOPENING_ELIGIBILITY.md`
- `79_GATE11A_NAMED_OPERATOR_PRESSURE_ADMISSIBILITY.md`
- `80_GATE11A_NAMED_OPERATOR_PRESSURE_ADMISSIBILITY_SMOKE.md`
- `81_GATE11B_BOUNDED_LINE_INSUFFICIENCY_DECLARABILITY.md`
- `82_GATE11B_BOUNDED_LINE_INSUFFICIENCY_DECLARABILITY_SMOKE.md`
- `83_GATE11C_BOUNDED_LINE_INSUFFICIENCY_DECLARATION_SURFACE_AUDIT.md`
- `84_GATE11C_BOUNDED_LINE_INSUFFICIENCY_DECLARATION_SURFACE_AUDIT_SMOKE.md`
- `85_GATE11D_ONE_BOUNDED_LINE_INSUFFICIENCY_EXPLICIT_DECLARATION_AUDIT.md`
- `86_GATE11D_ONE_BOUNDED_LINE_INSUFFICIENCY_EXPLICIT_DECLARATION_AUDIT_SMOKE.md`

The frozen Gate10 closeout-support line remains recorded in:

- `74_GATE10F_PRE_CLOSEOUT_JUDGMENT.md`
- `75_GATE10F_PRE_CLOSEOUT_JUDGMENT_SMOKE.md`

The first Gate11E explicit-declaration instantiation path audit consumer now exists in:

- `tools/run_gate11e_explicit_declaration_instantiation_path_audit.py`

The first tracked Gate11E explicit-declaration instantiation path smoke read is now recorded in:

- `88_GATE11E_EXPLICIT_DECLARATION_INSTANTIATION_PATH_AUDIT_SMOKE.md`

The next narrow Gate11F later-source instantiation admissibility audit slice is now tracked in:

- `89_GATE11F_LATER_SOURCE_INSTANTIATION_ADMISSIBILITY_AUDIT.md`

## 0. Scope

Gate11E is the fifth narrow Gate11 slice.

Gate11E does:

- ask what the minimum admissible later-source path would be from `surface_defined but not_yet_declared` to one future honest explicit declaration
- preserve the Gate10 closeout sentence exactly as already earned
- preserve the Gate11A absence result exactly as already recorded
- preserve the Gate11C `surface_defined` result exactly as already recorded
- preserve the Gate11D `not_yet_declared` result exactly as already recorded
- define only the minimum same-source additions by which one explicit bounded-line insufficiency declaration could later become honestly instantiated

Gate11E does not:

- declare a bounded-line insufficiency candidate
- re-audit whether a declaration already exists
- decide reopening eligibility
- reopen operator admission
- declare graph-wide operator behavior earned
- settle broader trusted-tree settlement
- retroactively reinterpret Gate9, Gate10, Gate11A, Gate11B, Gate11C, or Gate11D memory
- mine the repo outside the controlling source run
- choose one candidate out of multiple hypothetical futures

## 1. Controlling Source Run

Gate11E consumes exactly this controlling source run:

- `runs/gate11d_one_bounded_line_insufficiency_explicit_declaration_audit_smoke_from_gate11c`

No additional source run is in scope.

Under the currently frozen Gate11D source, the recorded upstream result is:

- `gate11c_declaration_surface_preservation_status = preserved`
- `bounded_line_insufficiency_explicit_declaration_marker_status = absent`
- `bounded_line_insufficiency_candidate_id_singularity_status = absent`
- `bounded_line_insufficiency_class_singularity_status = none`
- `bounded_line_host_failure_statement_status = absent`
- `one_bounded_line_insufficiency_explicit_declaration_status = not_yet_declared`
- `next_named_blocker = no_explicit_declaration_marker`

So Gate11E must treat the current controlling source as:

- a surface-defined source
- a not-yet-declared source
- a path-definition source only

The worker must not:

- infer a candidate identity from broader repo history
- supply a missing class by interpretation
- synthesize a host-failure statement from rhetorical pressure
- distribute a later declaration across multiple sources

## 2. Public Question

The Gate11E question is:

- `what is the minimum admissible path from surface_defined but not_yet_declared to one future honest explicit declaration?`

This is narrower than:

- candidate declaration itself
- explicit-declaration existence audit
- reopening-eligibility judgment
- operator reopening

It is only:

- the path-definition gate for the minimum later-source additions required before one explicit bounded-line insufficiency declaration could be audited honestly

## 3. Why Gate11E Exists

Gate11A earned:

- the correct current result is `absent / none / not_yet_admissible`

Gate11B earned:

- the correct current result is `absent / none / not_yet_declarable`

Gate11C earned:

- the future declaration surface is fixed
- the correct current result is `surface_defined`

Gate11D earned:

- the fixed Gate11C surface is preserved
- the correct current result is `not_yet_declared`
- the current blocker is missing instantiation, not missing surface

So the next honest move is not:

- invent a candidate
- relitigate declaration existence
- declare reopening eligible

It is:

- fix the minimum later-source path by which one explicit declaration could later become instantiated honestly

## 4. Path Discipline

Gate11E must define path, not declaration.

One future explicit declaration path counts as admissibly defined only if all of the following are fixed narrowly:

- the currently missing declaration components are named explicitly
- the minimum later-source instantiation rule is defined on a same-source basis
- the anti-shortcut boundary is explicit

If any of those conditions remains undefined, Gate11E must not treat later declaration as path-ready.

## 5. Missing Component Naming

Gate11E must name what is currently absent, not invent what the future candidate is.

Under the current frozen Gate11D source, the minimum missing declaration components are:

- no explicit declaration marker is instantiated
- no candidate id is instantiated
- no single class is instantiated
- no explicit host-failure statement is instantiated

Gate11E may summarize the current blocker compactly, but it may not:

- fill in the missing candidate id
- fill in the missing class
- fill in the missing host-failure sentence

## 6. Minimum Later-Source Instantiation Rule

Gate11E does not ask what the candidate should be.

It asks what the minimum same-source additions would have to be in one later frozen run before a later Gate11 explicit-declaration audit could honestly return `declared`.

That minimum later-source rule requires all of the following in the same later frozen source run:

- one explicit declaration marker instantiated under the Gate11C surface
- one and only one `bounded_line_insufficiency_candidate_id`
- one and only one class from the Gate11B class set attached to that same candidate id
- one explicit host-failure sentence stating `the current bounded line cannot honestly host <candidate_id>`
- one status payload, one registry row, and one read sentence all keyed to that same single candidate id

The minimum later-source rule therefore forbids:

- splitting declaration components across multiple runs
- one source for candidate id and another source for host failure
- one source for class and another source for declaration marker
- worker-side stitching of partial later-source fragments

## 7. Anti-Shortcut Boundary

Gate11E must keep the path narrow and non-promotional.

The later-source path may not be defined in a way that depends on:

- broader trusted-tree settlement promotion
- retroactive rewrite
- graph-wide leap
- worker-side synthesis

So the minimum later-source path remains admissible only if:

- broader trusted-tree settlement stays explicitly unearned
- operator admission stays denied
- retroactive reinterpretation stays forbidden
- the later declaration could be audited from one frozen source without worker-side completion

If the path becomes legible only by depending on shortcut moves, Gate11E must treat it as:

- `denied`

## 8. Required Judgment Checks

Gate11E may define the minimum later-source path only if all of the following remain clear.

### 8.1 Gate10 Closeout Preservation

The instantiation-path audit is invalid if:

- the Gate10 closeout sentence no longer remains preserved as recorded

### 8.2 Gate11A Absence Result Preservation

The instantiation-path audit is invalid if:

- the Gate11A absence result no longer remains preserved as recorded

### 8.3 Gate11C Surface Definition Preservation

The instantiation-path audit is invalid if:

- the Gate11C declaration surface is no longer preserved as `surface_defined`

### 8.4 Gate11D Not-Yet-Declared Preservation

The instantiation-path audit is invalid if:

- the Gate11D `not_yet_declared` result no longer remains preserved as recorded

### 8.5 Missing Component Naming Is Explicit

The instantiation-path audit is invalid if:

- the current missing declaration components are not named explicitly

### 8.6 Minimum Later-Source Rule Is Same-Source

The instantiation-path audit is invalid if:

- the path would permit declaration components to be assembled across multiple later sources

### 8.7 Single-Candidate And Single-Class Discipline Remains Fixed

The instantiation-path audit is invalid if:

- the path would permit multiple candidate ids or unresolved class ambiguity

### 8.8 Host-Failure Statement Requirement Remains Explicit

The instantiation-path audit is invalid if:

- the path would permit later declaration without an explicit host-failure sentence for the same candidate id

### 8.9 Anti-Shortcut Boundary Remains Intact

The instantiation-path audit is invalid if:

- the path depends on broader-settlement promotion, retroactive rewrite, graph-wide leap, or worker-side synthesis

## 9. Falsifiers

Gate11E fails path definition if any of the following happens:

- Gate10 closeout preservation fails
- Gate11A absence-result preservation fails
- Gate11C surface-defined preservation fails
- Gate11D not-yet-declared preservation fails
- the missing declaration components are not named explicitly
- the later-source rule would require multiple runs
- the later-source rule would permit multiple candidate ids
- the later-source rule would permit multiple unresolved classes
- the host-failure requirement is left implicit
- broader-settlement inflation is needed
- retroactive reinterpretation pressure appears
- graph-wide operator leap pressure appears
- worker-side synthesis would be required

## 10. Expected Outputs

Any Gate11E implementation must emit exactly these files:

- `manifest.json`
- `checksums.json`
- `explicit_declaration_instantiation_path_registry.jsonl`
- `explicit_declaration_instantiation_path_policy_compare.csv`
- `explicit_declaration_instantiation_path_status.json`
- `gate11e_explicit_declaration_instantiation_path_read.md`

## 11. Required Status Keys

Any Gate11E implementation must emit explicit status for:

- `gate10_closeout_preservation_status`
- `gate11a_absence_result_preservation_status`
- `gate11c_declaration_surface_preservation_status`
- `gate11d_not_yet_declared_state_preservation_status`
- `broader_trusted_tree_settlement_still_unearned_status`
- `operator_admission_still_denied_status`
- `retroactive_reinterpretation_forbidden_status`
- `missing_surface_component_naming_status`
- `minimal_later_source_instantiation_rule_status`
- `anti_shortcut_boundary_status`
- `explicit_declaration_instantiation_path_status`
- `next_named_blocker`

## 12. Status Space

Gate11E is limited to the following judgment space.

### 12.1 Preservation Statuses

Each of:

- `gate10_closeout_preservation_status`
- `gate11a_absence_result_preservation_status`
- `gate11c_declaration_surface_preservation_status`
- `gate11d_not_yet_declared_state_preservation_status`

must be emitted as one of:

- `preserved`
- `not_preserved`

### 12.2 Boundary Statuses

Each of:

- `broader_trusted_tree_settlement_still_unearned_status`
- `operator_admission_still_denied_status`
- `retroactive_reinterpretation_forbidden_status`
- `anti_shortcut_boundary_status`

must be emitted as one of:

- `confirmed`
- `not_confirmed`
- `denied`
- `deferred`

### 12.3 Missing Component Naming Status

`missing_surface_component_naming_status` must be emitted as one of:

- `named`
- `not_yet_named`
- `denied`
- `deferred`

### 12.4 Minimum Later-Source Instantiation Rule Status

`minimal_later_source_instantiation_rule_status` must be emitted as one of:

- `defined`
- `not_yet_defined`
- `denied`
- `deferred`

### 12.5 Path Outcome Status

`explicit_declaration_instantiation_path_status` must be emitted as one of:

- `path_defined`
- `not_yet_defined`
- `denied`
- `deferred`

`path_defined` means only:

- the minimum later-source path by which one explicit bounded-line insufficiency declaration could honestly become instantiated is now fixed

It does not mean:

- a candidate is declared
- reopening is eligible
- operator reopening is earned

## 13. Outcome Ladder

Gate11E outcomes are limited to these four.

### 13.1 Path Defined

Use `path_defined` only if:

- the currently missing declaration components are named explicitly
- the minimum later-source instantiation rule is defined on a same-source basis
- the anti-shortcut boundary is explicit
- no falsifier fires

### 13.2 Not Yet Defined

Use `not_yet_defined` if:

- the line remains preserved through Gate11D
- but the minimum later-source path is not yet fixed narrowly enough to audit one future explicit declaration honestly

### 13.3 Denied

Use `denied` if:

- the path depends on shortcut, inflation, rewrite, graph-wide leap, or worker-side synthesis

### 13.4 Deferred

Use `deferred` if:

- the frozen source evidence is incomplete or contradictory in a way that blocks path judgment

## 14. Forbidden

The following remain forbidden in Gate11E:

- no candidate declaration
- no explicit-declaration existence judgment
- no reopening-eligibility judgment
- no operator reopening
- no graph-wide operator
- no broader trusted-tree settlement promotion
- no retroactive rewrite of Gate9, Gate10, Gate11A, Gate11B, Gate11C, or Gate11D
- no repo-wide mining outside the controlling source run
- no worker-side synthesis in place of later-source instantiation
- no multiple-candidate resolution
- no scalar comeback
- no benchmark-zoo expansion
- no sheaf branding
- no higher-gauge branding
- no KAGAMI rhetoric inside the public verdict

## 15. Delegation Boundary

An implementation worker may do:

- narrow consumer implementation
- unit tests
- status payload emission
- tracked smoke handoff formatting

An implementation worker may not do:

- candidate invention
- explicit-declaration judgment redesign
- reopening-eligibility judgment
- operator reopening judgment
- doctrine redesign
- blocker naming redesign
- falsifier redesign
- scope widening
- theory branding

If the spec is insufficient, the work must stop and report the gap rather than invent behavior.

## 16. Memory Hook

The Gate11E sentence is:

- Gate11E does not declare a candidate
- it fixes the minimum later-source path by which one explicit bounded-line insufficiency declaration could honestly become instantiated

The shortest acceptable memory hook is:

- `Gate11E does not declare a candidate; it fixes the minimum same-source additions by which one explicit bounded-line insufficiency declaration could later become honestly instantiated.`
