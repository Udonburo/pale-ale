# Gate11A Named Operator-Pressure Admissibility

Status: first implementation landed and first smoke execution recorded
Role: named operator-pressure admissibility audit, not operator reopening or reopening-eligibility judgment
Date: 2026-03-23

Gate11A proceeds from:

- `76_GATE10_CLOSEOUT.md`
- `78_GATE11_OPERATOR_REOPENING_ELIGIBILITY.md`

The frozen Gate10 closeout-support line remains recorded in:

- `74_GATE10F_PRE_CLOSEOUT_JUDGMENT.md`
- `75_GATE10F_PRE_CLOSEOUT_JUDGMENT_SMOKE.md`

The first Gate11A named operator-pressure admissibility consumer now exists in:

- `tools/run_gate11a_named_operator_pressure_admissibility.py`

The first tracked Gate11A admissibility smoke read is now recorded in:

- `80_GATE11A_NAMED_OPERATOR_PRESSURE_ADMISSIBILITY_SMOKE.md`

## 0. Scope

Gate11A is the first narrow Gate11 slice.

Gate11A does:

- ask whether any real named operator-pressure case exists at all under the frozen post-Gate10 line
- preserve the bounded Gate10 closeout sentence
- preserve broader trusted-tree settlement as still unearned
- preserve operator admission as still denied

Gate11A does not:

- decide operator-reopening eligibility
- reopen operator admission
- declare graph-wide operator behavior earned
- settle broader trusted-tree settlement
- retroactively reinterpret Gate9 or Gate10 memory
- invent a pressure case that is not explicitly present in the frozen source

## 1. Controlling Source Run

Gate11A consumes exactly this controlling source run:

- `runs/gate10f_pre_closeout_judgment_smoke_from_gate10e`

No additional source run is in scope.

The worker must not:

- mine earlier Gate10 bundles directly
- mine Gate9 docs directly
- use repo-wide historical text to fabricate a new pressure case

## 2. Public Question

The Gate11A question is:

- `does any admissible named operator-pressure case exist at all under the frozen post-Gate10 line?`

This is narrower than:

- operator-reopening eligibility
- operator reopening
- graph-wide operator retry

It is only:

- the admissibility gate for whether a real named pressure case exists

## 3. Why Gate11A Exists

Gate10 closed with:

- three declared narrow slice-local `settled` results
- bounded broader-pattern support
- bounded closeout sentence support
- broader trusted-tree settlement still unearned
- operator admission still denied

So the first honest Gate11 move is not:

- reopen the operator line

It is:

- ask whether there is any real named pressure case at all beyond the frozen Gate10 boundary

## 4. Admissibility Discipline

Gate11A must use a stricter standard than rhetorical dissatisfaction.

A named operator-pressure case counts only if all of the following are true:

- it is explicitly named in the controlling source run
- it is legible as an object-level or artifact-level case
- it falls into at least one admissible class declared in `78_GATE11_OPERATOR_REOPENING_ELIGIBILITY.md`
- it does not depend on broader-settlement inflation, retroactive rewrite, or graph-wide leap pressure

If the controlling source run contains no such explicit named case, Gate11A must treat the case as:

- absent

In that default absence case:

- `admissible_pressure_class_status` must be `none`
- `named_operator_pressure_admissibility_status` must default to `not_yet_admissible` unless a separate `denied` or `deferred` condition independently fires

Absence may not be replaced by:

- interpretive extrapolation
- slogan completion
- historical mining from out-of-scope files

## 5. Required Judgment Checks

Gate11A may recognize an admissible named pressure case only if all of the following remain clear.

### 5.1 Gate10 Closeout Preservation

The admissibility audit is invalid if:

- the Gate10 closeout sentence no longer remains preserved as recorded

### 5.2 Bounded Closeout-Support Preservation

The admissibility audit is invalid if:

- the Gate10F bounded closeout-support line no longer remains preserved as recorded

### 5.3 Broader Settlement Still Unearned

The admissibility audit is invalid if:

- the pressure case depends on promoting broader trusted-tree settlement into something already earned

### 5.4 Operator Admission Still Denied

The admissibility audit is invalid if:

- the pressure case assumes operator reopening instead of testing whether a named pressure exists

### 5.5 Retroactive Reinterpretation Remains Forbidden

The admissibility audit is invalid if:

- the pressure case appears only after rewriting Gate9 or Gate10 memory

### 5.6 Named Pressure Case Is Explicit

The admissibility audit is invalid if:

- the pressure case is not explicitly named in the controlling source run

### 5.7 Bounded-Line Insufficiency Evidence Is Explicit

The admissibility audit is invalid if:

- the source names a discomfort but does not show that the current bounded line is actually insufficient to host it honestly

## 6. Falsifiers

Gate11A fails admissibility if any of the following happens:

- Gate10 closeout preservation fails
- bounded closeout-support preservation fails
- broader-settlement inflation is needed
- operator reopening is assumed rather than audited
- retroactive reinterpretation pressure appears
- no explicit named operator-pressure case exists in the controlling source
- bounded-line insufficiency evidence is absent
- graph-wide operator leap pressure appears

## 7. Expected Outputs

Any Gate11A implementation must emit exactly these files:

- `manifest.json`
- `checksums.json`
- `named_operator_pressure_admissibility_registry.jsonl`
- `named_operator_pressure_admissibility_policy_compare.csv`
- `named_operator_pressure_admissibility_status.json`
- `gate11a_named_operator_pressure_admissibility_read.md`

## 8. Required Status Keys

Any Gate11A implementation must emit explicit status for:

- `gate10_closeout_preservation_status`
- `bounded_closeout_support_preservation_status`
- `broader_trusted_tree_settlement_still_unearned_status`
- `operator_admission_still_denied_status`
- `retroactive_reinterpretation_forbidden_status`
- `named_operator_pressure_case_status`
- `admissible_pressure_class_status`
- `bounded_line_insufficiency_evidence_status`
- `graph_wide_operator_leap_pressure_status`
- `named_operator_pressure_admissibility_status`
- `next_named_blocker`

## 9. Status Space

Gate11A is limited to the following judgment space.

### 9.1 Preservation Statuses

Each of:

- `gate10_closeout_preservation_status`
- `bounded_closeout_support_preservation_status`

must be emitted as one of:

- `preserved`
- `not_preserved`

### 9.2 Boundary Statuses

Each of:

- `broader_trusted_tree_settlement_still_unearned_status`
- `operator_admission_still_denied_status`
- `retroactive_reinterpretation_forbidden_status`

must be emitted as one of:

- `confirmed`
- `not_confirmed`

### 9.3 Named Pressure Presence Status

`named_operator_pressure_case_status` must be emitted as one of:

- `present`
- `absent`
- `deferred`

### 9.4 Admissible Pressure Class Status

`admissible_pressure_class_status` must be emitted as one of:

- `tree_choice_instability`
- `current_bounded_line_insufficiency`
- `nonlocal_reconciliation_pressure`
- `narrow_reopening_pressure_without_graph_wide_leap`
- `none`
- `deferred`

### 9.5 Insufficiency Evidence Status

`bounded_line_insufficiency_evidence_status` must be emitted as one of:

- `present`
- `absent`
- `deferred`

### 9.6 Graph-Wide Leap Pressure Status

`graph_wide_operator_leap_pressure_status` must be emitted as one of:

- `absent`
- `present`

### 9.7 Admissibility Outcome Status

`named_operator_pressure_admissibility_status` must be emitted as one of:

- `admissible`
- `not_yet_admissible`
- `denied`
- `deferred`

`admissible` means only:

- a real named pressure case exists and may be used in a later reopening-eligibility slice

It does not mean:

- operator reopening is eligible
- operator reopening is earned

## 10. Outcome Ladder

Gate11A outcomes are limited to these four.

### 10.1 Admissible

Use `admissible` only if:

- a named operator-pressure case is explicitly present in the controlling source
- the case falls into at least one admissible class
- bounded-line insufficiency evidence is present
- no falsifier fires

### 10.2 Not Yet Admissible

Use `not_yet_admissible` if:

- the frozen post-Gate10 line does not yet contain a real admissible named pressure case
- but denial by inflation or rewrite is not yet the right verdict

### 10.3 Denied

Use `denied` if:

- the pressure case depends on inflation, retroactive rewrite, or graph-wide leap pressure

### 10.4 Deferred

Use `deferred` if:

- the frozen source evidence is incomplete or contradictory in a way that blocks admissibility judgment

## 11. Forbidden

The following remain forbidden in Gate11A:

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

## 12. Delegation Boundary

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

## 13. Memory Hook

The Gate11A sentence is:

- Gate11A asks only whether any real named operator-pressure case exists at all under the frozen post-Gate10 line

The shortest acceptable memory hook is:

- `Gate11A does not ask whether reopening is eligible yet; it asks whether there is any admissible named pressure case at all.`
