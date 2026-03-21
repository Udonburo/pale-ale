# Gate9Q Post-Adoption Integration

Status: spec-only, implementation not yet landed
Role: Gate9Q post-adoption integration / mainline summary update spec, not operator opening or broader trusted-tree settlement
Date: 2026-03-22

Gate9Q proceeds from:

- `58_GATE9P_DECLARED_SPLIT_ADOPT_OR_DEFER_JUDGMENT.md`
- `59_GATE9P_DECLARED_SPLIT_ADOPT_OR_DEFER_JUDGMENT_SMOKE.md`

## 0. Why This Exists

Gate9P earned one narrow judgment result.

What is now known is:

- the Gate9N declared split is adopted on a forward basis into the Gate9 mainline audit lane
- operator admission remains denied
- broader trusted-tree settlement remains unresolved
- prior Gate9 reads are not retroactively reinterpreted

What is not yet fixed is:

- how that forward-basis adoption should be integrated into the Gate9 mainline memory and summary documents

So the next honest move is not:

- reopen operator admission
- widen the tree-line claim
- rewrite earlier Gate9 results as if they had always used the adopted split

It is:

- specifying post-adoption integration on a forward-only basis

## 1. Scope

Gate9Q studies only:

- how to integrate the forward-basis Gate9P adopt judgment into Gate9 mainline memory
- which summary surfaces should now mention the adopted split
- which unresolved boundaries must remain explicit after that integration

It does not:

- reopen adopt-or-defer judgment
- reopen adoption-worthiness
- change the trusted-edge policy
- change the first forest
- reopen operator admission
- settle the broader trusted-tree line
- introduce new metrics or new public roles

## 2. Fixed Conditions

Gate9Q keeps all of the following fixed:

- forward-basis adoption fixed
- operator admission still denied
- broader trusted-tree settlement still unresolved
- no retroactive reinterpretation of earlier Gate9 reads
- declared split definition fixed as `closure_return_leg_auxiliary` / `residual_chord_candidate`

This matters because Gate9Q is an integration step, not a hidden escalation step.

## 3. Public Question

The Gate9Q question is:

- `how should the forward-basis adopted split be integrated into Gate9 mainline memory without reopening operator admission or retroactively rewriting earlier Gate9 lines?`

This question is intentionally narrow.

It does not ask whether more of the trusted-tree line should now be promoted.

It asks only how the already-earned forward-basis adoption should be reflected in mainline memory.

## 4. Public Integration Surface

Gate9Q may update only:

- Gate9 mainline memory
- forward pointers between Gate9 stage documents
- summary sentences that describe the current forward basis

Gate9Q must not use integration as a license to update:

- operator status
- broader tree-line verdicts
- earlier Gate9 scientific judgments

## 5. Source Run

The frozen source run for Gate9Q is:

- `runs/gate9p_declared_split_adopt_or_defer_smoke_from_gate9o`

If that source run is unavailable or inconsistent with the current spec, implementation must stop at gap reporting.

## 6. Expected Output Files

Gate9Q implementation should emit exactly these first-class outputs:

- `manifest.json`
- `post_adoption_integration_registry.jsonl`
- `post_adoption_integration_policy_compare.csv`
- `post_adoption_integration_status.json`
- `gate9q_post_adoption_integration_read.md`

Additional helper files are allowed only if they do not change the public object.

## 7. Required Status Keys

The Gate9Q status payload must emit at least:

- `forward_basis_adoption_status`
- `mainline_memory_update_status`
- `operator_admission_still_denied_status`
- `retroactive_reinterpretation_forbidden_status`
- `broader_tree_settlement_unresolved_status`
- `historical_lane_preservation_status`
- `integration_scope_preservation_status`
- `post_adoption_integration_readiness_status`
- `integration_outcome_status`
- `next_named_blocker`

These keys are required so integration remains explicit and bounded.

## 8. Falsifiers

Gate9Q must keep these falsifiers explicit:

- integration requires retroactive reinterpretation of prior Gate9 reads
- integration implies operator admission is now open
- integration implies broader trusted-tree settlement
- integration widens doctrine beyond the declared forward-basis split
- integration cannot state what remains unresolved

If those falsifiers land, post-adoption integration has not earned execution.

## 9. Forbidden

The Gate9Q implementation must not:

- reopen operator admission
- settle the broader trusted-tree line
- rename blockers
- rename falsifiers
- widen scope
- introduce new metrics
- introduce new public roles
- change doctrine
- silently convert forward-basis adoption into retroactive policy

If the spec appears insufficient, implementation must:

- emit the gap clearly
- stop without filling it by invention

## 10. What This Spec Can Earn

At most, Gate9Q can earn the right to say:

- the forward-basis adoption has now been cleanly integrated into Gate9 mainline memory

It still does not earn:

- operator opening
- broader trusted-tree settlement
- retroactive reinterpretation of earlier Gate9 lines
- a new geometry claim

## 11. Opus-Safe Delegation Boundary

Gate9Q is safe to delegate only as:

- narrow consumer implementation
- unit tests
- status payload emission
- tracked smoke handoff formatting

Gate9Q is not delegated as:

- blocker naming
- falsifier design
- scope narrowing
- doctrine change
- broader workstream judgment beyond post-adoption integration

## 12. Current Memory Hook

The shortest acceptable sentence is:

- integrate the adoption forward, not backward
