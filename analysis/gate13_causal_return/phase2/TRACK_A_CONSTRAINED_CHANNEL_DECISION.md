# Track A constrained-channel decision

Status fixed on 2026-08-20:

- `TRACK_A_FREE_GENERATION_CHANNEL = TERMINATED_INSTRUMENT_CHANNEL_INADEQUATE`
- Reason: free-form outputs did not satisfy the frozen serialization grammar, so A0 was never opened and no semantic capability judgment was supported.
- `TRACK_A_SCIENTIFIC_QUESTION = OPEN`
- `TRACK_A_CONSTRAINED_REGISTER_CHANNEL = AUTHORIZED_FOR_BOUNDED_REDESIGN_AND_EXECUTION`
- `A0 = UNOPENED`, `A1 = UNOPENED`, `A2 = UNOPENED`

The new variant inherits the existing Track A cases, prompts, register algebra, oracle, metrics, thresholds, runtime, and conditional ladder. Its sole scientific-design change is replacement of free generation by a syntax-only finite-state output channel. Every register-value and answer-value slot independently permits both `0` and `1`; the channel does not enforce XOR transitions, oracle truth, internal consistency, or equality between the final register and answer.

M1 is development/preflight only. Its responses are isolated from A0/A1/A2 checkpoints and metrics. The prior two Modal executions, their authorizations and results, and all five consumed case-level forwards remain immutable.

A3, Track C, formal Gate13, activation extraction, B2a repair, and B2b execution remain closed.
