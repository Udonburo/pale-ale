# Gate11G Later-Source Naming Surface Audit

Status: first implementation landed and first smoke execution recorded
Role: later-source naming surface audit, not later-source admissibility itself, candidate declaration, explicit-declaration existence judgment, reopening-eligibility judgment, or operator reopening
Date: 2026-03-23

Gate11G proceeds from:

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
- `87_GATE11E_EXPLICIT_DECLARATION_INSTANTIATION_PATH_AUDIT.md`
- `88_GATE11E_EXPLICIT_DECLARATION_INSTANTIATION_PATH_AUDIT_SMOKE.md`
- `89_GATE11F_LATER_SOURCE_INSTANTIATION_ADMISSIBILITY_AUDIT.md`
- `90_GATE11F_LATER_SOURCE_INSTANTIATION_ADMISSIBILITY_AUDIT_SMOKE.md`

The frozen Gate10 closeout-support line remains recorded in:

- `74_GATE10F_PRE_CLOSEOUT_JUDGMENT.md`
- `75_GATE10F_PRE_CLOSEOUT_JUDGMENT_SMOKE.md`

The first Gate11G later-source naming surface audit consumer now exists in:

- `tools/run_gate11g_later_source_naming_surface_audit.py`

The first tracked Gate11G later-source naming surface smoke read is now recorded in:

- `92_GATE11G_LATER_SOURCE_NAMING_SURFACE_AUDIT_SMOKE.md`

The next narrow Gate11H one later-source explicit-naming audit slice is now tracked in:

- `93_GATE11H_ONE_LATER_SOURCE_EXPLICIT_NAMING_AUDIT.md`

## 0. Scope

Gate11G is the seventh narrow Gate11 slice.

Gate11G does:

- ask what would count as a valid explicit naming surface for one later source to carry the fixed Gate11E path into a later explicit-declaration audit
- preserve the Gate10 closeout sentence exactly as already earned
- preserve the Gate11A absence result exactly as already recorded
- preserve the Gate11C `surface_defined` result exactly as already recorded
- preserve the Gate11D `not_yet_declared` result exactly as already recorded
- preserve the Gate11E `path_defined` result exactly as already recorded
- preserve the Gate11F `not_yet_admissible` result exactly as already recorded
- define only the narrow explicit naming surface by which one later source could later count as named

Gate11G does not:

- decide whether one later source is already admissible
- declare a bounded-line insufficiency candidate
- declare that explicit declaration already exists
- decide reopening eligibility
- reopen operator admission
- declare graph-wide operator behavior earned
- settle broader trusted-tree settlement
- retroactively reinterpret Gate9, Gate10, Gate11A, Gate11B, Gate11C, Gate11D, Gate11E, or Gate11F memory
- mine the repo outside the controlling source run
- choose a later source worker-side

## 1. Controlling Source Run

Gate11G consumes exactly this controlling source run:

- `runs/gate11f_later_source_instantiation_admissibility_audit_smoke_from_gate11e`

No additional source run is in scope.

Under the currently frozen Gate11F source, the recorded upstream result is:

- `gate11e_path_defined_state_preservation_status = preserved`
- `later_source_naming_status = absent`
- `later_source_cardinality_status = none`
- `same_source_path_attachment_status = not_attached`
- `anti_shortcut_boundary_status = confirmed`
- `later_source_instantiation_admissibility_status = not_yet_admissible`

So Gate11G must treat the current controlling source as:

- a path-preserved but later-source-unnamed source
- not a later-source-admissible source

The worker must not:

- invent a later source identifier
- infer naming from narrative future-language
- infer naming from later-source admissibility wishes
- resolve later-source ambiguity by worker-side selection

## 2. Public Question

The Gate11G question is:

- `what would count as a valid explicit naming surface for one later source to carry the fixed Gate11E path into a later explicit-declaration audit?`

This is narrower than:

- later-source admissibility itself
- candidate declaration itself
- explicit-declaration existence judgment
- reopening-eligibility judgment
- operator reopening

It is only:

- the surface-definition gate for what would count as an explicit later-source naming surface

## 3. Why Gate11G Exists

Gate11F earned:

- the fixed Gate11E path remains preserved
- the correct current result is `not_yet_admissible`
- no one later source is yet named or attached

So the next honest move is not:

- invent a later source
- declare later-source admissibility already earned
- declare a candidate
- declare reopening eligible

It is:

- define what would count as an explicit later-source naming surface before any later-source admissibility judgment is attempted again

## 4. Later-Source Naming Surface Discipline

Gate11G must define surface conditions, not later-source admissibility itself.

One later source counts as explicitly named only if all of the following are fixed:

- an explicit later-source marker shape exists
- later-source singularity is fixed at one source per run
- full-path attachment shape is fixed on that same later source
- no shortcut or worker-side synthesis is required

If those conditions are not fixed, Gate11G must not promote the line into later-source naming surface definition.

## 5. Required Surface Conditions

### 5.1 Explicit Later-Source Marker Shape

One later source counts as explicitly named only if the surface requires:

- an explicit `later_source_id` or `later_frozen_run_id`
- an explicit naming marker that is auditable in status, registry, or read form

The following do not count:

- narrative future-language
- generic later-run wishes
- worker-side inference

### 5.2 Single Later-Source Singularity

One later source counts as explicitly named only if the surface requires:

- exactly one later source in one run
- no second competing later source

Multiple later sources must be treated as:

- `deferred`

### 5.3 Full-Path Attachment Shape

One later source counts as explicitly named only if the surface requires that the same later source carry:

- one declaration marker
- one candidate id
- one class
- one explicit host-failure sentence
- matched status, registry, and read surfaces

If those elements are not fixed onto the same later source, naming surface is not yet defined.

### 5.4 Anti-Shortcut Boundary

One later source counts as explicitly named only if the surface forbids reliance on:

- broader-settlement promotion
- retroactive rewrite
- graph-wide leap
- worker-side synthesis

## 6. Required Judgment Checks

Gate11G may recognize a later-source naming surface only if all of the following remain clear.

### 6.1 Gate10 Closeout Preservation

The later-source naming surface audit is invalid if:

- the Gate10 closeout sentence no longer remains preserved as recorded

### 6.2 Gate11A Absence Result Preservation

The later-source naming surface audit is invalid if:

- the Gate11A absence result no longer remains preserved as recorded

### 6.3 Gate11C Surface Definition Preservation

The later-source naming surface audit is invalid if:

- the Gate11C declaration surface is no longer preserved as `surface_defined`

### 6.4 Gate11D Not-Yet-Declared Preservation

The later-source naming surface audit is invalid if:

- the Gate11D `not_yet_declared` result no longer remains preserved as recorded

### 6.5 Gate11E Path-Defined Preservation

The later-source naming surface audit is invalid if:

- the Gate11E `path_defined` result no longer remains preserved as recorded

### 6.6 Gate11F Not-Yet-Admissible Preservation

The later-source naming surface audit is invalid if:

- the Gate11F `not_yet_admissible` result no longer remains preserved as recorded

### 6.7 Explicit Later-Source Marker Shape Is Fixed

The later-source naming surface audit is invalid if:

- the controlling source does not fix what explicit later-source marker would count as naming

### 6.8 Single Later-Source Singularity Is Fixed

The later-source naming surface audit is invalid if:

- the controlling source does not fix one-source singularity

### 6.9 Full-Path Attachment Shape Is Fixed

The later-source naming surface audit is invalid if:

- the controlling source does not fix how the full Gate11E path attaches to the same later source

### 6.10 Anti-Shortcut Boundary Remains Intact

The later-source naming surface audit is invalid if:

- the naming surface depends on broader-settlement promotion, retroactive rewrite, graph-wide leap, or worker-side synthesis

## 7. Falsifiers

Gate11G fails later-source naming surface definition if any of the following happens:

- Gate10 closeout preservation fails
- Gate11A absence-result preservation fails
- Gate11C surface-defined preservation fails
- Gate11D not-yet-declared preservation fails
- Gate11E path-defined preservation fails
- Gate11F not-yet-admissible preservation fails
- explicit later-source marker shape is not fixed
- later-source singularity is not fixed
- full-path attachment shape is not fixed
- broader-settlement inflation is needed
- retroactive reinterpretation pressure appears
- graph-wide operator leap pressure appears
- worker-side synthesis would be required

## 8. Expected Outputs

Any Gate11G implementation must emit exactly these files:

- `manifest.json`
- `checksums.json`
- `later_source_naming_surface_registry.jsonl`
- `later_source_naming_surface_policy_compare.csv`
- `later_source_naming_surface_status.json`
- `gate11g_later_source_naming_surface_read.md`

## 9. Required Status Keys

Any Gate11G implementation must emit explicit status for:

- `gate10_closeout_preservation_status`
- `gate11a_absence_result_preservation_status`
- `gate11c_declaration_surface_preservation_status`
- `gate11d_not_yet_declared_state_preservation_status`
- `gate11e_path_defined_state_preservation_status`
- `gate11f_not_yet_admissible_state_preservation_status`
- `broader_trusted_tree_settlement_still_unearned_status`
- `operator_admission_still_denied_status`
- `retroactive_reinterpretation_forbidden_status`
- `explicit_later_source_marker_shape_status`
- `single_later_source_singularity_status`
- `full_path_attachment_shape_status`
- `anti_shortcut_boundary_status`
- `later_source_naming_surface_status`
- `next_named_blocker`

## 10. Status Space

Gate11G is limited to the following judgment space.

### 10.1 Preservation Statuses

Each of:

- `gate10_closeout_preservation_status`
- `gate11a_absence_result_preservation_status`
- `gate11c_declaration_surface_preservation_status`
- `gate11d_not_yet_declared_state_preservation_status`
- `gate11e_path_defined_state_preservation_status`
- `gate11f_not_yet_admissible_state_preservation_status`

must be emitted as one of:

- `preserved`
- `not_preserved`

### 10.2 Boundary Statuses

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

### 10.3 Explicit Later-Source Marker Shape Status

`explicit_later_source_marker_shape_status` must be emitted as one of:

- `defined`
- `not_defined`
- `deferred`

### 10.4 Single Later-Source Singularity Status

`single_later_source_singularity_status` must be emitted as one of:

- `defined`
- `not_defined`
- `deferred`

### 10.5 Full-Path Attachment Shape Status

`full_path_attachment_shape_status` must be emitted as one of:

- `defined`
- `not_defined`
- `deferred`

### 10.6 Later-Source Naming Surface Outcome Status

`later_source_naming_surface_status` must be emitted as one of:

- `surface_defined`
- `not_yet_defined`
- `denied`
- `deferred`

`surface_defined` means only:

- the later-source naming surface is now fixed narrowly enough to audit later

It does not mean:

- one later source is already admissible
- a candidate is declared
- explicit declaration already exists
- reopening is eligible

## 11. Outcome Ladder

Gate11G outcomes are limited to these four.

### 11.1 Surface Defined

Use `surface_defined` only if:

- explicit later-source marker shape is fixed
- single later-source singularity is fixed
- full-path attachment shape is fixed
- no falsifier fires

### 11.2 Not Yet Defined

Use `not_yet_defined` if:

- the line remains preserved through Gate11F
- but the later-source naming surface is not yet fixed narrowly enough

### 11.3 Denied

Use `denied` if:

- the later-source naming surface depends on shortcut, inflation, rewrite, graph-wide leap, or worker-side synthesis

### 11.4 Deferred

Use `deferred` if:

- the frozen source evidence is incomplete or contradictory in a way that blocks later-source naming surface judgment

## 12. Forbidden

The following remain forbidden in Gate11G:

- no later-source admissibility judgment
- no candidate declaration
- no explicit-declaration existence judgment
- no reopening-eligibility judgment
- no operator reopening
- no graph-wide operator
- no broader trusted-tree settlement promotion
- no retroactive rewrite of Gate9, Gate10, Gate11A, Gate11B, Gate11C, Gate11D, Gate11E, or Gate11F
- no repo-wide mining outside the controlling source run
- no worker-side synthesis in place of later-source naming
- no multiple-source resolution
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

- later-source invention
- later-source admissibility judgment redesign
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

## 14. Memory Hook

The Gate11G sentence is:

- Gate11G does not admit a later source
- it asks what would count as an explicit naming surface for one later source to carry the fixed Gate11E path

The shortest acceptable memory hook is:

- `Gate11G does not admit a later source; it defines what would count as an explicit naming surface for one later source to carry the fixed Gate11E path into a later explicit-declaration audit.`
