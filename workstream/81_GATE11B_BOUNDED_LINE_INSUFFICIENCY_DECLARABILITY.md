# Gate11B Bounded-Line Insufficiency Declarability

Status: first implementation landed and first smoke execution recorded
Role: bounded-line insufficiency declarability absence-preservation audit, not reopening-eligibility judgment or operator reopening
Date: 2026-03-23

Gate11B proceeds from:

- `76_GATE10_CLOSEOUT.md`
- `78_GATE11_OPERATOR_REOPENING_ELIGIBILITY.md`
- `79_GATE11A_NAMED_OPERATOR_PRESSURE_ADMISSIBILITY.md`
- `80_GATE11A_NAMED_OPERATOR_PRESSURE_ADMISSIBILITY_SMOKE.md`

The frozen Gate10 closeout-support line remains recorded in:

- `74_GATE10F_PRE_CLOSEOUT_JUDGMENT.md`
- `75_GATE10F_PRE_CLOSEOUT_JUDGMENT_SMOKE.md`

The first Gate11B bounded-line insufficiency declarability consumer now exists in:

- `tools/run_gate11b_bounded_line_insufficiency_declarability.py`

The first tracked Gate11B bounded-line insufficiency declarability smoke read is now recorded in:

- `82_GATE11B_BOUNDED_LINE_INSUFFICIENCY_DECLARABILITY_SMOKE.md`

## 0. Scope

Gate11B is the second narrow Gate11 slice.

Gate11B does:

- ask whether any post-Gate10 pressure can be honestly named as a bounded-line insufficiency
- preserve the Gate10 closeout sentence exactly as already earned
- preserve the Gate11A absence result exactly as already recorded
- decide only whether one bounded-line insufficiency candidate may be declared honestly
- preserve the current no-candidate state when the frozen controlling source contains no explicit declaration surface

Gate11B does not:

- decide reopening eligibility
- reopen operator admission
- declare graph-wide operator behavior earned
- settle broader trusted-tree settlement
- retroactively reinterpret Gate9 or Gate10 memory
- mine the repo outside the controlling source run
- declare more than one insufficiency candidate at a time

## 1. Controlling Source Run

Gate11B consumes exactly this controlling source run:

- `runs/gate11a_named_operator_pressure_admissibility_smoke_from_gate10f`

No additional source run is in scope.

Gate11B is not a declaration-generation slice.

Under the currently frozen Gate11A source, the recorded upstream result is:

- `named_operator_pressure_case_status = absent`
- `admissible_pressure_class_status = none`
- `named_operator_pressure_admissibility_status = not_yet_admissible`

So Gate11B must treat the current controlling source as an absence-preservation source unless one explicit bounded-line insufficiency declaration is already present inside that same frozen run.

The worker must not:

- mine earlier Gate10 bundles directly
- mine Gate9 docs directly
- use repo-wide historical text to fabricate a new insufficiency declaration
- choose one candidate out of multiple undeclared possibilities

## 2. Public Question

The Gate11B question is:

- `can any post-Gate10 pressure be honestly named as a bounded-line insufficiency without relying on settlement inflation, retroactive rewrite, or graph-wide leap pressure?`

This is narrower than:

- reopening-eligibility judgment
- operator reopening
- graph-wide operator retry

It is only:

- the declarability gate for whether one bounded-line insufficiency candidate may be named honestly
- and, under the current frozen source, the absence-preservation gate for whether the line still contains no such explicit candidate

## 3. Why Gate11B Exists

Gate11A earned:

- the post-Gate10 line has now been checked for a real named operator-pressure case
- the current frozen Gate10F source contains no such explicit named case
- the correct current result is `absent / none / not_yet_admissible`

So the next honest move is not:

- declare reopening eligible
- reopen the operator line

It is:

- ask whether one bounded-line insufficiency candidate can even be declared honestly under the frozen post-Gate10 line
- and preserve `absent / none / not_yet_declarable` if the frozen source still contains no explicit declaration surface

## 4. Declarability Discipline

Gate11B must use a stricter standard than speculative theory preference.

A bounded-line insufficiency candidate counts as declarable only if all of the following are true:

- it is explicitly declared in the controlling source run
- exactly one candidate is declared
- it is legible as a bounded-line insufficiency rather than a generic dissatisfaction
- it does not depend on settlement inflation, retroactive rewrite, or graph-wide leap pressure

If the controlling source run contains no such explicit declaration, Gate11B must treat the candidate as:

- absent

In that default absence case:

- `bounded_line_insufficiency_candidate_status` must be `absent`
- `bounded_line_insufficiency_class_status` must be `none`
- `bounded_line_insufficiency_declarability_status` must default to `not_yet_declarable` unless a separate `denied` or `deferred` condition independently fires

If the controlling source run contains more than one distinct explicit candidate declaration, Gate11B must treat the slice as:

- `deferred`

Under the current frozen Gate11A source, the default absence path is expected unless a later tracked source supersedes this frozen input.

Absence or multiplicity may not be replaced by:

- interpretive extrapolation
- rhetorical completion
- repo-wide mining from out-of-scope files
- worker-side candidate selection

## 5. Bounded-Line Insufficiency Classes

Gate11B may treat a candidate as classifiable only if it falls into exactly one of the following classes.

### 5.1 Tree-Choice Instability

Use this only if:

- the candidate names instability in trusted-tree / residual-chord partition choice within declared topology hygiene

### 5.2 Current Bounded-Line Insufficiency

Use this only if:

- the candidate names a direct insufficiency in the current bounded Gate10 line itself

### 5.3 Nonlocal Reconciliation Pressure

Use this only if:

- the candidate names a cross-slice reconciliation pressure that the current bounded line cannot honestly host

### 5.4 Narrow Reopening Pressure Without Graph-Wide Leap

Use this only if:

- the candidate names a genuinely narrow reopening-shape pressure
- and that pressure does not require a graph-wide leap

## 6. Required Judgment Checks

Gate11B may declare one bounded-line insufficiency candidate only if all of the following remain clear.

### 6.1 Gate10 Closeout Preservation

The declarability audit is invalid if:

- the Gate10 closeout sentence no longer remains preserved as recorded

### 6.2 Gate11A Absence Result Preservation

The declarability audit is invalid if:

- the Gate11A absence result no longer remains preserved as recorded

### 6.3 Broader Settlement Still Unearned

The declarability audit is invalid if:

- the candidate depends on promoting broader trusted-tree settlement into something already earned

### 6.4 Operator Admission Still Denied

The declarability audit is invalid if:

- the candidate assumes operator reopening instead of staying inside named bounded-line insufficiency language

### 6.5 Retroactive Reinterpretation Remains Forbidden

The declarability audit is invalid if:

- the candidate appears only after rewriting Gate9 or Gate10 memory

### 6.6 Candidate Declaration Is Explicit

The declarability audit is invalid if:

- the candidate is not explicitly declared in the controlling source run

### 6.7 Candidate Scope Is Singular

The declarability audit is invalid if:

- more than one distinct candidate is declared in the controlling source run

### 6.8 Graph-Wide Leap Pressure Is Absent

The declarability audit is invalid if:

- the candidate only makes sense by reintroducing graph-wide operator behavior, smoothing, or field-style rescue

## 7. Falsifiers

Gate11B fails declarability if any of the following happens:

- Gate10 closeout preservation fails
- Gate11A absence-result preservation fails
- broader-settlement inflation is needed
- operator reopening is assumed rather than deferred
- retroactive reinterpretation pressure appears
- no explicit bounded-line insufficiency candidate exists in the controlling source
- more than one distinct candidate is declared
- graph-wide operator leap pressure appears

## 8. Expected Outputs

Any Gate11B implementation must emit exactly these files:

- `manifest.json`
- `checksums.json`
- `bounded_line_insufficiency_declarability_registry.jsonl`
- `bounded_line_insufficiency_declarability_policy_compare.csv`
- `bounded_line_insufficiency_declarability_status.json`
- `gate11b_bounded_line_insufficiency_declarability_read.md`

## 9. Required Status Keys

Any Gate11B implementation must emit explicit status for:

- `gate10_closeout_preservation_status`
- `gate11a_absence_result_preservation_status`
- `broader_trusted_tree_settlement_still_unearned_status`
- `operator_admission_still_denied_status`
- `retroactive_reinterpretation_forbidden_status`
- `bounded_line_insufficiency_candidate_status`
- `bounded_line_insufficiency_class_status`
- `settlement_inflation_pressure_status`
- `graph_wide_operator_leap_pressure_status`
- `bounded_line_insufficiency_declarability_status`
- `next_named_blocker`

## 10. Status Space

Gate11B is limited to the following judgment space.

### 10.1 Preservation Statuses

Each of:

- `gate10_closeout_preservation_status`
- `gate11a_absence_result_preservation_status`

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

### 10.3 Candidate Presence Status

`bounded_line_insufficiency_candidate_status` must be emitted as one of:

- `present`
- `absent`
- `deferred`

### 10.4 Candidate Class Status

`bounded_line_insufficiency_class_status` must be emitted as one of:

- `tree_choice_instability`
- `current_bounded_line_insufficiency`
- `nonlocal_reconciliation_pressure`
- `narrow_reopening_pressure_without_graph_wide_leap`
- `none`
- `deferred`

### 10.5 Inflation Pressure Status

`settlement_inflation_pressure_status` must be emitted as one of:

- `absent`
- `present`

### 10.6 Graph-Wide Leap Pressure Status

`graph_wide_operator_leap_pressure_status` must be emitted as one of:

- `absent`
- `present`

### 10.7 Declarability Outcome Status

`bounded_line_insufficiency_declarability_status` must be emitted as one of:

- `declarable`
- `not_yet_declarable`
- `denied`
- `deferred`

`declarable` means only:

- one bounded-line insufficiency candidate may now be named honestly in a later slice

It does not mean:

- reopening is eligible
- operator reopening is earned

Under the current frozen Gate11A source, `declarable` is not expected to be a live path unless that same frozen run already contains one explicit bounded-line insufficiency declaration marker.

## 11. Outcome Ladder

Gate11B outcomes are limited to these four.

### 11.1 Declarable

Use `declarable` only if:

- one bounded-line insufficiency candidate is explicitly present in the controlling source
- exactly one class applies
- no falsifier fires

This path is unavailable under the current frozen Gate11A source unless an explicit declaration marker is already present in that same run.

### 11.2 Not Yet Declarable

Use `not_yet_declarable` if:

- the frozen post-Gate10 line does not yet contain a single explicit bounded-line insufficiency declaration
- but denial by inflation, rewrite, or graph-wide leap is not yet the right verdict

This is the expected default result under the current frozen Gate11A source.

### 11.3 Denied

Use `denied` if:

- the candidate depends on settlement inflation, retroactive rewrite, or graph-wide leap pressure

### 11.4 Deferred

Use `deferred` if:

- the frozen source evidence is incomplete or contradictory in a way that blocks declarability judgment
- or more than one distinct candidate is declared and worker-side selection would be required

## 12. Forbidden

The following remain forbidden in Gate11B:

- no reopening-eligibility judgment
- no operator reopening
- no graph-wide operator
- no broader trusted-tree settlement promotion
- no retroactive rewrite of Gate9 or Gate10
- no repo-wide mining outside the controlling source run
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

- reopening-eligibility judgment
- operator reopening judgment
- doctrine redesign
- blocker naming redesign
- falsifier redesign
- scope widening
- theory branding

If the spec is insufficient, the work must stop and report the gap rather than invent behavior.

## 14. Memory Hook

The Gate11B sentence is:

- Gate11B asks only whether one bounded-line insufficiency candidate can be named honestly under the frozen post-Gate10 line
- and preserves absence when the frozen Gate11A source still contains no explicit declaration surface

The shortest acceptable memory hook is:

- `Gate11B does not ask whether reopening is eligible; it asks whether even one bounded-line insufficiency candidate can be declared honestly, and otherwise preserves not-yet-declarable absence.`
