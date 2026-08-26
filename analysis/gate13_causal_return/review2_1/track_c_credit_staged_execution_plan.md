# Track C Review 2.1 credit-staged execution contract

This is an operational forecast, not execution authority. Modal balance,
billing refresh date, price, runtime availability, and authorization must be
verified manually before every paid stage. Credits are not assumed to roll
over.

## Pre-stage campaign freeze

Before any Stage M forward, create and validate one immutable campaign manifest
covering all 20 blocks and every possible map/behavior case. Required frozen
items are the exact Qwen3.6-27B and tokenizer revision, runtime image and
dependencies, chat-template bytes/hash, score slot and single-token checks,
block/map/behavior IDs, `natural_rule_v1`, fresh codebooks/demonstrations/seeds,
map halves, depth allocation, deterministic P/Q render ledgers, randomized
block-interleaved orders, qualification rules, exact resume rules, analysis
seed, and the permutation schedule family. This pre-stage step does not call a
model.

Surface validation is a pre-forward hard gate. Any P/Q mismatch closes Review
2.1 before Stage M. No matched no-op fallback has been authorized.

## Stage M — map collection

```text
20 blocks × 2 halves × 5 nodes × 24 samples = 4,800 forwards
historical variable-rate forecast                  = USD 20.49952640
historical acquisition/preflight allowance         = USD  0.12368778
Stage M planning total                             = USD 20.62321418
```

Stage M runs only the frozen case IDs in the frozen order. It collects the
map-side inputs already needed for operators, validity gates, and `C_b^M`.
`C_b^M` uses the required exact-node logits and causes no extra forward.

After all accepted Stage M IDs are complete, outcome-blind qualification runs
once. It requires at least 16 blocks and at least four at every depth, plus all
rank, conditioning, leverage, permutation-support, `R_b` variance, split-half,
frame-rank, exact-square reproducibility, broken-square sensitivity,
path-surface, and artifact-completeness gates. No block is replaced.

```text
MAP_COMPLETE_NOT_QUALIFIED
  -> terminal close; Stage E remains forbidden

MAP_COMPLETE_AND_QUALIFIED
  -> Stage E becomes eligible, not authorized
```

## Optional sealed billing wait

If eligible but the manually verified balance or billing date does not support
Stage E, enter:

```text
SEALED_WAIT_FOR_BILLING_CYCLE
```

No behavior outcome exists in this state and no design change is permitted.
Human-visible information is limited to qualification state, qualified counts
by depth, artifact hashes, forward accounting, and billing/resume state. The
map predictor values and packet summaries remain sealed. Hash verification and
backup are allowed; scientific inspection is not.

Before leaving the wait, a human must manually recheck the actual Modal balance,
refresh date, runtime availability, forecast, USD 65 ceiling, and separate
execution authority. A billing refresh grants no authority by itself.

## Stage E — behavior collection

Stage E opens only after `MAP_COMPLETE_AND_QUALIFIED` and a separate human paid
stage decision. Each qualified depth-`d` block has
`2 paths × 24 episodes × (d + 1)` frozen forwards. The pre-frozen full order is
filtered to the qualified block IDs without reordering.

| Qualified design | Stage E forwards | Stage E forecast | Campaign forwards | Campaign expected forecast |
|---|---:|---:|---:|---:|
| 16 blocks, 4/depth | 4,608 | USD 19.67954534 | 9,408 | USD 40.30275952 |
| 20 blocks, 5/depth | 5,760 | USD 24.59943168 | 10,560 | USD 45.22264586 |

Intermediate qualified counts use their exact per-depth composition, never an
episode-average approximation. The frozen historical rate is USD
`0.004270734666666666` per forward.

The full-design planning references remain:

```text
expected forecast:     USD 45.22  (exact historical-linear value 45.22264586)
25% contingency:       USD 56.53  (exact 56.528307325)
hard planning ceiling: USD 65.00
```

These are forecasts, not guaranteed charges, credit promises, budget
allocations, or authorization.

## Exact resume and completion

```text
EXACT_RESUME_BEHAVIOR
  -> run only frozen case IDs that have no accepted artifact

TRACK_C_COMPLETE
  -> verify completeness, analyze once under the SHA-locked pipeline, and stop
```

Every accepted case ID is immutable. It may not be rerun, duplicated, replaced,
or given a new seed because of content, latency, failure rate, billing timing,
or any scientific result. A missing ID may be resumed only with its original
payload and position in the filtered frozen order. Provider/runtime drift or a
forecast above the ceiling is a stop condition, not permission to reduce or
redesign the experiment.

Allowed operational transitions are:

```text
FROZEN_CAMPAIGN
  -> Stage M
  -> MAP_COMPLETE_NOT_QUALIFIED -> terminal close
  -> MAP_COMPLETE_AND_QUALIFIED
       -> Stage E
       -> SEALED_WAIT_FOR_BILLING_CYCLE
            -> EXACT_RESUME_BEHAVIOR
  -> TRACK_C_COMPLETE -> analyze once -> stop
```

No operational state opens A3, Formal Gate13, hidden intervention, or Track C
without a separate human authorization.
