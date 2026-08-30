# Gate13 Candidate — Track B Post-Phase-2 Decision

## Scope

This is a tracked state correction after the frozen Phase 2 source-sufficiency
stop. It does not amend either Phase 2 lock, reopen the historical 12-run B2a
substrate, authorize fresh B2b execution, or open Track C or formal Gate13.

## Binding evidence

```text
planning_base_commit            5245ff00fd06730f71e2239ef0f30aee3e79d0e1
review1_snapshot_commit         2eaea5d7a94885151a8b3c170a43e4051820948e
phase2_code_commit              259884d5dc146877bb95428c987697a17a6fbd22
handoff_sha256                  7981c6339d6b7d4e1a07fc2b1e5c092bed119d96e0dfa8c998599adad3f35bcd
review1_report_sha256           0401ce462cb8bbf03cf0013c68c2ccec9dcf73d4db64b212af02f45014c0ad20
phase2_a_lock_sha256            9c4b94b5199c3d355e8707798ba9bc1797aa2d690762b226f57bec63742215fa
phase2_b2a_lock_sha256          58317d5a608a6b6717e189c142cfafce47fdb42d63ace734f5e32e50ef27d714
phase2_dual_authorization_sha256 150717d4922b5a471393faf49353d0775898e762a03e6025a7e1bf6166f14d5c
phase2_final_state_sha256       e4b173273fe056d412a813f5d918084dae0de06ee83c6c89411bb7e7bc717e63
b2a_result_sha256               da8a611b94f6c0104e0d66af4961e43b061235b9d633f6d5a5f78ba2ad2cb14c
```

The source-sufficiency probe found no retained independent sample-half source
in any of the 12 bound historical runs. No B2a operator outcome row was read,
and no reconstruction, stability, or scalar-shadow stage opened.

## Decision

```text
B2A_HISTORICAL_12RUN
  = TERMINATED_SUBSTRATE_INADEQUATE
  reason = independent sample-half source unavailable in 12/12 runs

TRACK_B_SCIENTIFIC_QUESTION
  = OPEN

B2B_FRESH_SUBSTRATE
  = RESERVED_NOT_AUTHORIZED

TRACK_A
  = READY_FOR_EXTERNAL_EXECUTION

A3
  = CLOSED

TRACK_C
  = CLOSED

FORMAL_GATE13
  = CLOSED
```

`TERMINATED_SUBSTRATE_INADEQUATE` closes only the historical 12-run substrate.
It is not a negative answer to the Track B scientific question. Fresh B2b is a
reservation for an independently constructed substrate, not an activation or
execution authorization.

Track A remains independent. Its effect size, failure location, and prompt
differences must not be used to choose B metrics. On this branch, an A2 PASS may
serve only as a resource gate for a separately authorized fresh B2b. An A2 FAIL
would end the current causal-register coupling; it would not refute Track B in
general.

Activation extraction, B2b execution, A3, Track C, and formal Gate13 remain
closed. No new constitution, general cycle-space definition, or public README
change is authorized by this decision.
