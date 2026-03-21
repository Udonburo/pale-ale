# Gate9P Declared Split Adopt-Or-Defer Judgment

Status: spec-only, implementation not yet landed
Role: Gate9P adopt-or-defer judgment spec, not operator opening or broader trusted-tree settlement
Date: 2026-03-22

Gate9P proceeds from:

- `56_GATE9O_DECLARED_SPLIT_ADOPTION_WORTHINESS.md`
- `57_GATE9O_DECLARED_SPLIT_ADOPTION_WORTHINESS_SMOKE.md`

## 0. Why This Exists

Gate9O earned one narrow but serious result.

What is now known is:

- the Gate9N declared split is adoption-worthy under the frozen Gate9 doctrine
- bypass readiness changes from denied to clear under that split
- conflict-side bridge preservation remains clear
- closure doctrine remains clear
- operator admission still does not open

What is not yet known is:

- whether that adoption-worthy split should actually be adopted into the Gate9 mainline audit lane
- or whether it should remain audit-only despite its adoption-worthiness

So the next honest move is not:

- treat adoption-worthiness as automatic adoption
- reopen operator admission
- inflate the result into a full mainline success claim

It is:

- making an explicit adopt-or-defer judgment

## 1. Scope

Gate9P studies only:

- whether the Gate9N declared split should be adopted into the Gate9 mainline audit lane on a forward basis
- or deferred despite being adoption-worthy

It does not:

- reopen role-coupling separability
- reopen adoption-worthiness
- change the trusted-edge policy
- change the first forest
- change the cycle family
- reopen operator admission
- introduce new metrics or new public roles

## 2. Fixed Conditions

Gate9P keeps all of the following fixed:

- trusted-edge policy fixed
- first forest fixed
- closure doctrine fixed
- operator admission still denied
- declared split definition fixed as `closure_return_leg_auxiliary` / `residual_chord_candidate`

This matters because Gate9P is a judgment step, not a hidden redesign step.

## 3. Public Question

The Gate9P question is:

- `should the Gate9N declared split be adopted into the Gate9 mainline audit lane, or deferred despite adoption-worthiness?`

This question is intentionally narrow.

It does not ask whether the full trusted-tree line is settled.

It asks only whether the declared split has earned forward mainline use within the Gate9 audit lane.

## 4. Public Comparison

Gate9P must compare only:

- `adopt_candidate`
- `defer_candidate`

The comparison is not about score superiority.

It is about judgment discipline:

- does adoption preserve mainline comparability
- does adoption preserve the audit-lane / operator boundary
- does adoption avoid hidden doctrine widening
- is there still any named reason to defer

## 5. Source Run

The frozen source run for Gate9P is:

- `runs/gate9o_declared_split_adoption_worthiness_smoke_from_gate9n`

If that source run is unavailable or inconsistent with the current spec, implementation must stop at gap reporting.

## 6. Expected Output Files

Gate9P implementation should emit exactly these first-class outputs:

- `manifest.json`
- `declared_split_adopt_or_defer_registry.jsonl`
- `declared_split_adopt_or_defer_policy_compare.csv`
- `declared_split_adopt_or_defer_status.json`
- `gate9p_declared_split_adopt_or_defer_read.md`

Additional helper files are allowed only if they do not change the public object.

## 7. Required Status Keys

The Gate9P status payload must emit at least:

- `adoption_worthiness_status`
- `mainline_comparability_preservation_status`
- `audit_lane_boundary_preservation_status`
- `operator_admission_non_promotion_status`
- `historical_reinterpretation_required_status`
- `doctrine_scope_change_required_status`
- `adopt_candidate_status`
- `defer_candidate_status`
- `judgment_outcome_status`
- `next_named_blocker`

These keys are required so the judgment stays explicit rather than drifting into informal endorsement language.

## 8. Falsifiers

Gate9P must keep these falsifiers explicit:

- adopting the split would require historical reinterpretation of prior Gate9 reads rather than a forward-only policy change
- adopting the split would widen doctrine beyond the declared role split
- adopting the split would weaken the audit-lane / operator boundary
- adopting the split would require hidden role surgery or bundle-specific exception logic
- deferral still has a named blocker that survives Gate9O

If the adoption-side falsifiers land, the split has not earned adoption.

If the deferral-side blocker does not exist, deferral must not be treated as cost-free caution.

## 9. Forbidden

The Gate9P implementation must not:

- declare operator admission open
- rename blockers
- rename falsifiers
- widen scope
- introduce new metrics
- introduce new public roles
- change doctrine
- silently convert adopt-or-defer judgment into full mainline rollout

If the spec appears insufficient, implementation must:

- emit the gap clearly
- stop without filling it by invention

## 10. What This Spec Can Earn

At most, Gate9P can earn the right to say one of:

- the declared split is adopted into the Gate9 mainline audit lane on a forward basis
- the declared split remains audit-only for one named reason

It still does not earn:

- operator opening
- broader trusted-tree settlement
- retroactive reinterpretation of earlier Gate9 lines
- a new geometry claim

## 11. Opus-Safe Delegation Boundary

Gate9P is safe to delegate only as:

- narrow consumer implementation
- unit tests
- status payload emission
- tracked smoke handoff formatting

Gate9P is not delegated as:

- blocker naming
- falsifier design
- scope narrowing
- doctrine change
- broader workstream judgment beyond adopt-or-defer

## 12. Current Memory Hook

The shortest acceptable sentence is:

- judge adopt-or-defer explicitly before treating adoption-worthiness as policy
