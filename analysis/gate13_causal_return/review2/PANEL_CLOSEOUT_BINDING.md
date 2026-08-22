# Gate13 Track C — Review 2 panel closeout binding

Date: 2026-08-23

This package closes only the **Review 2 estimand-design phase**. It does not
authorize or perform Track C scientific execution. The terminal Review 2 state
is:

```text
REVIEW2_READY_FOR_HUMAN_AUTHORIZATION
```

Track C, A3, and formal Gate13 remain closed. No Modal API was called; no GPU
was allocated; no model was downloaded or loaded; no model forward or
activation collection was performed; and no Track C coupling outcome was
inspected. A separate, explicit human authorization would be required before
any future execution. This package is not that authorization.

## Immutable incoming state

```text
PANEL_A2_AND_B_PASS = immutable terminal campaign state

VISIBLE_STATE_USE
  Qwen3.5-27B = QUALIFIED
  Qwen3.6-27B = QUALIFIED
  Qwen3-8B    = UNTESTED_DUE_TO_UPSTREAM_A1_FAIL
  Qwen3-14B   = UNTESTED_DUE_TO_UPSTREAM_A0_FAIL
  Qwen3.8-27B = UNTESTED_DUE_TO_UPSTREAM_A0_FAIL

FRESH_SQUARE_OPERATOR_INSTRUMENT / Qwen3.6-27B = QUALIFIED
COUPLED_COUNTERFACTUAL_NATURALITY               = UNTESTED
HIDDEN_CAUSAL_REGISTER                          = UNTESTED
A3 / TRACK_C / FORMAL_GATE13                    = CLOSED
```

Nothing in Review 2 amends those results. Qualification on one or more models
is not a uniqueness result, and visible-state use is not evidence for a hidden
causal register.

## Frozen target and candidate design

- Exact model target: `Qwen/Qwen3.6-27B` at revision
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- Experimental unit: one independent fresh naturality-square block.
- Proposed design: 20 planned blocks, 24 map samples per node per half, 24
  behavior episodes per block, and five blocks at each rollout depth
  2/4/6/8.
- Fail-closed analysis minimum: 16 qualified blocks and at least four qualified
  blocks at every rollout depth. Failed blocks are not replaced and blocks are
  never added after outcome observation.
- Primary test: one representation feature, one behavioral outcome, and one
  nuisance-preserving block-level permutation test of the leave-one-block-out
  relative held-out SSE reduction.

The full estimand and all gates are frozen in
`track_c_estimand_lock_candidate.json`; the human-readable protocol is
`TRACK_C_REVIEW2_PROTOCOL.md`.

## Review 2 evidence boundary

The sample-size study read only the already-collected Qwen3.6 fresh-B
activation NPZ files. It did not read response/logit files and did not use a
Track C outcome. The 8/12/16/24 grid and the model-free sensitivity simulation
made zero model calls.

The only empirically eligible map cloud size was 24 samples per node per half.
The proposed design forecasts 4,800 map forwards plus 5,760 behavioral
forwards, or 10,560 future scientific forwards. Historical linear extrapolation
is USD 45.22264586; a frozen 25% contingency gives USD 56.528307325 under a
declared future hard ceiling of USD 65.00. These numbers are planning bounds,
not a GPU reservation or spending authority.

## Provenance binding

| Object | SHA-256 |
|---|---|
| Base Git commit | `eb494cc24df627f5807c2aee8a8ab3717393b17c` |
| Checkpoint transfer panel lock | `27972f4ba4920c45b272fa7ea6360cbae2fb4cc748a1ef9ededa681a5dad8526` |
| Fresh-square reservations | `22f875050a16a0ad0f170539cdf99bc145fc555c5908ba647c632fb4a86d9e24` |
| Panel execution authorization | `ed71079bf9905ce3cec5a1d6ecd7c515cbda0f3464d86e4bbb69db1358a745f5` |
| Campaign terminal state | `da299a8c695531c741a17870698d458090ac722bbb3bd4fdba4b0f5563c8b75d` |
| Panel Track A results | `4e1eac855dfd1622afba3f7edaff98295bb2caac8982aff88c841fecb4d02413` |
| Billing summary | `851685f178cb84f128ad7e2ca5eafdd2f6acff38a65d98ae0bfb241b4d3ad6fc` |
| Qwen3.6 execution claim | `53d6c77b74bc73ca2a7abfc2b88cb6158316095dbbe3497ee712c999187ed804` |
| Fresh-B terminal state | `bfc7bdf083ee1f49ab3efb87fda98966cdcc5f1a08c50612d02ec5fdf441e34b` |
| Fresh-B qualification result | `3edb6ce5bedb7c9ec2df93a319b25c8253010460fb14375949ae2869f7bae464` |
| Fresh-B artifact manifest | `e0c07028f8d0bad5beb74a19d12e374e461f031bc1fde4ba65ac6593a4f17fec` |
| Fresh-B manifest payload | `4be617978a27718d6c76a1bee5a3a23b5b1a358aff5f062490740d2842f0caa1` |
| Fresh-B collection ledger | `839da5097d14144188840ccb56c8d08d22fd8a97a418f2106f804755d0a4720e` |

Any mismatch in the exact revision, frozen template, layer set, provenance, or
authority state is a provenance blocker, not permission to repair or continue.

## Validation snapshot

```text
python analysis/gate13_causal_return/review2/track_c_review2_validator.py
  PASS / 0 errors

python -m unittest discover \
  -s analysis/gate13_causal_return/review2/tests \
  -p 'test_*.py' -v
  8 passed / 0 failed
```

The targeted suite covers gauge invariance, block-level behavior aggregation,
exact LOBO/PRESS equivalence, deterministic nuisance-preserving permutation,
forward/cost accounting, existing-B downsampling reproduction, JSON locks, and
the complete fail-closed package validator.

## Mandatory boundary

Raw operator overlap remains representation geometry, not a causal transition.
Review 2 makes no first-use claim for naturality, representation holonomy, or
intermediate-state edits. It also does not open Track C, create an execution
authorization, or permit an outcome-adaptive redesign.
