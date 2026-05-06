# Gate12B Archive Strict-Support Sensitivity Memo

Status: strict-support sensitivity memo draft
Role: bounded Gate12B archive-family sensitivity check over existing dense-transformer Gate12A artifacts, not a checkpoint revision, not a release claim, not an invariant-law promotion, and not a Gate12A schema change
Date: 2026-05-06

This memo proceeds from:

- `195_GATE11_ADMISSIBILITY_MEMBRANE_FREEZE.md`
- `196_GATE12_DISCRETE_CONNECTION_CONSTITUTION.md`
- `197_GATE12A_DISCRETE_CONNECTION_IMPLEMENTATION_CONTRACT.md`
- `198_GATE12A_DISCRETE_CONNECTION_AUDIT.md`
- `217_GATE12B_OBSERVER_RELATIVE_COARSE_GRAINED_CLOSURE_OPENING_MEMO.md`
- `218_GATE12B_FIRST_OBSERVER_RELATIVE_COARSE_GRAINED_CLOSURE_SMOKE_MEMO.md`
- `219_GATE12B_QWEN_2_5_0_5B_THREE_FAMILY_OBSERVER_RELATIVE_COMPARISON_MEMO.md`
- `220_GATE12B_DENSE_TRANSFORMER_FAMILY_EFFECT_EXPANSION_MEMO.md`

## 0. Scope

This memo records a strict-support sensitivity check for the archive-family
Gate12B signal from `220`.

It keeps the family fixed to:

- `archive_v1`

and checks the current dense-transformer archive surfaces:

- `Qwen/Qwen2.5-0.5B`
- `Qwen/Qwen2.5-3B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`
- `Qwen/Qwen3-4B`

The purpose is to ask whether the archive relation-signature flip from `220`
survives stricter candidate support rules.

It does not:

- claim a universal law
- change Gate12A artifacts
- change Gate12B runner semantics
- add a release surface
- include sidecar or non-mainline architecture rows

## 1. Runs

The following sensitivity settings were run:

- baseline reference: `top_k = 3`, `min_observer_support = 2`, `min_scale_support = 2`
- stricter scale support: `top_k = 1 / 3 / 5`, `min_observer_support = 2`, `min_scale_support = 3`
- stricter observer support: `top_k = 3`, `min_observer_support = 3`, `min_scale_support = 2`
- strictest tested combination: `top_k = 1 / 5`, `min_observer_support = 3`, `min_scale_support = 3`

All runs are read-only secondary audits over existing Gate12A archive artifacts.
All runs use CPU-only local artifact reads.

## 2. Baseline Reference

The `220` baseline archive rows were:

| model | total candidates | high | flat | dominant high signature | dominant flat signature |
| --- | ---: | ---: | ---: | --- | --- |
| `Qwen/Qwen2.5-0.5B` | 92 | 21 | 71 | `residual_chord=3` / 18 | `residual_chord=1|trusted_tree=2` / 68 |
| `Qwen/Qwen2.5-3B-Instruct` | 108 | 35 | 73 | `residual_chord=3` / 32 | `residual_chord=1|trusted_tree=2` / 70 |
| `meta-llama/Llama-3.2-3B-Instruct` | 114 | 37 | 77 | `residual_chord=3` / 34 | `residual_chord=1|trusted_tree=2` / 74 |
| `Qwen/Qwen3-4B` | 106 | 37 | 69 | `residual_chord=3` / 34 | `residual_chord=1|trusted_tree=2` / 66 |

This is the baseline archive flip:

- high-tension candidates concentrate on `residual_chord=3`
- flat candidates concentrate on `residual_chord=1|trusted_tree=2`

## 3. Strict Scale-Support Result

With `min_observer_support = 2` and `min_scale_support = 3`, the archive flip
survives at `top_k = 1`, `top_k = 3`, and `top_k = 5`.

### top_k = 1

| model | total | high | flat | high signature | flat signature |
| --- | ---: | ---: | ---: | --- | --- |
| `Qwen/Qwen2.5-0.5B` | 2 | 1 | 1 | `residual_chord=3` / 1 | `residual_chord=1|trusted_tree=2` / 1 |
| `Qwen/Qwen2.5-3B-Instruct` | 2 | 1 | 1 | `residual_chord=3` / 1 | `residual_chord=1|trusted_tree=2` / 1 |
| `meta-llama/Llama-3.2-3B-Instruct` | 2 | 1 | 1 | `residual_chord=3` / 1 | `residual_chord=1|trusted_tree=2` / 1 |
| `Qwen/Qwen3-4B` | 2 | 1 | 1 | `residual_chord=3` / 1 | `residual_chord=1|trusted_tree=2` / 1 |

### top_k = 3

| model | total | high | flat | high signature | flat signature |
| --- | ---: | ---: | ---: | --- | --- |
| `Qwen/Qwen2.5-0.5B` | 6 | 3 | 3 | `residual_chord=3` / 3 | `residual_chord=1|trusted_tree=2` / 3 |
| `Qwen/Qwen2.5-3B-Instruct` | 6 | 3 | 3 | `residual_chord=3` / 3 | `residual_chord=1|trusted_tree=2` / 3 |
| `meta-llama/Llama-3.2-3B-Instruct` | 6 | 3 | 3 | `residual_chord=3` / 3 | `residual_chord=1|trusted_tree=2` / 3 |
| `Qwen/Qwen3-4B` | 6 | 3 | 3 | `residual_chord=3` / 3 | `residual_chord=1|trusted_tree=2` / 3 |

### top_k = 5

| model | total | high | flat | high signature | flat signature |
| --- | ---: | ---: | ---: | --- | --- |
| `Qwen/Qwen2.5-0.5B` | 10 | 5 | 5 | `residual_chord=3` / 5 | `residual_chord=1|trusted_tree=2` / 5 |
| `Qwen/Qwen2.5-3B-Instruct` | 10 | 5 | 5 | `residual_chord=3` / 5 | `residual_chord=1|trusted_tree=2` / 5 |
| `meta-llama/Llama-3.2-3B-Instruct` | 10 | 5 | 5 | `residual_chord=3` / 5 | `residual_chord=1|trusted_tree=2` / 5 |
| `Qwen/Qwen3-4B` | 10 | 5 | 5 | `residual_chord=3` / 5 | `residual_chord=1|trusted_tree=2` / 5 |

This means the stricter scale-support path preserves the archive
relation-signature flip but no longer preserves flat-candidate dominance.
It becomes a balanced high/flat boundary read at the selected top-k values.

## 4. Strict Observer-Support Result

With `min_observer_support = 3`, candidates vanish across all four archive
surfaces:

| setting | total candidates per archive run |
| --- | ---: |
| `top_k = 3`, `min_observer_support = 3`, `min_scale_support = 2` | 0 |
| `top_k = 1`, `min_observer_support = 3`, `min_scale_support = 3` | 0 |
| `top_k = 5`, `min_observer_support = 3`, `min_scale_support = 3` | 0 |

This does not falsify the `220` archive flip.
It says the current candidate definition has only two independent observer-scope
supports available for the repeated archive flip under the tested views.

## 5. Gauge Boundary

All strict-support archive sensitivity runs had:

- `gauge_unstable_check_count = 0`

The maximum residual deltas stayed at floating-point scale:

- `Qwen/Qwen2.5-0.5B`: `4.440892098500626e-16`
- `Qwen/Qwen2.5-3B-Instruct`: `8.881784197001252e-16`
- `meta-llama/Llama-3.2-3B-Instruct`: `2.220446049250313e-16`
- `Qwen/Qwen3-4B`: `4.440892098500626e-16`

## 6. Reading Boundary

This memo earns the bounded sentence:

- the archive relation-signature flip survives stricter scale support across the current dense-transformer archive set and across `top_k = 1 / 3 / 5`

It does not earn:

- that flat-candidate dominance survives every stricter setting
- that three independent observer scopes support the current archive flip
- that the candidate rows are laws rather than candidate signals
- that the result should be mixed with sidecar-only architecture rows

## 7. Short Sentence

Under stricter Gate12B archive sensitivity, the candidate-dominance claim weakens
but the relation-signature flip strengthens as a bounded repeated signal:
`residual_chord=3` remains the high-tension side and
`residual_chord=1|trusted_tree=2` remains the flat side across all four current
dense-transformer archive surfaces for `top_k = 1 / 3 / 5` when scale support is
raised to three.
