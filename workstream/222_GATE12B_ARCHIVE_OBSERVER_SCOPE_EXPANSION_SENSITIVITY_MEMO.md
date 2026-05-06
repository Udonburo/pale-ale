# Gate12B Archive Observer-Scope Expansion Sensitivity Memo

Status: observer-scope expansion sensitivity memo draft
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
- `221_GATE12B_ARCHIVE_STRICT_SUPPORT_SENSITIVITY_MEMO.md`

## 0. Scope

This memo records an observer-scope expansion sensitivity check for the archive
Gate12B signal from `220` and `221`.

It keeps the family fixed to:

- `archive_v1`

and checks the current dense-transformer archive surfaces:

- `Qwen/Qwen2.5-0.5B`
- `Qwen/Qwen2.5-3B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`
- `Qwen/Qwen3-4B`

The purpose is to ask whether the archive relation-signature flip from `220`
can survive three independent observer scopes when the observer set is expanded
with artifact-native cycle-motif views.

It does not:

- change the default Gate12B observer mode set
- change Gate12A artifacts
- change Gate12A residual, transport, or holonomy semantics
- treat observer expansion as a law promotion
- add graph-wide smoothing or external physical-field semantics
- include sidecar or non-mainline architecture rows

## 1. Background

`221` found a useful asymmetry:

- strict scale support preserved the archive relation-signature flip
- strict observer support with the default `core_v1` observer set removed all
  candidates at `min_observer_support = 3`

The strict observer result came from available independent observer scopes, not
from a change in the archive relation pattern. Under `core_v1`, the archive
surface effectively had:

- `all_edges` and `anchor_qualified` as the same cycle-membership scope
- `residual_chord_heavy` as the residual-heavy scope
- `relation_kind_conditioned` as the relation-kind scope

That left only two independent scopes available for the repeated high/flat
archive flip.

## 2. Observer Expansion

This sensitivity check uses:

- `observer_mode_set = cycle_motif_expansion_v1`

The mode set keeps all `core_v1` observers and adds ordered cycle-motif views:

- `residual_first_leg`
- `residual_second_leg`
- `residual_third_leg`

These observers are derived from `ordered_relation_kind_path` and are still
read-only views over existing Gate12A explicit triangle artifacts.

They do not introduce new edge semantics.
They only ask whether a residual-chord position inside the explicit triangle
path gives an additional artifact-native observer scope.

For the archive runs, the observed independent scopes were:

| observer scope | observers |
| --- | --- |
| `observer_scope:000` | `all_edges`, `anchor_qualified` |
| `observer_scope:001` | `residual_chord_heavy`, `residual_second_leg` |
| `observer_scope:002` | `relation_kind_conditioned` |
| `observer_scope:003` | `residual_first_leg` |
| `observer_scope:004` | `residual_third_leg` |

## 3. Runs

All runs used:

- `observer_mode_set = cycle_motif_expansion_v1`
- `min_observer_support = 3`
- `min_scale_support = 3`
- `top_k = 1 / 3 / 5`

The output directories are:

- `runs/gate12b_observer_relative_coarse_grained_closure_qwen2_5_0_5b_archive_motif_obs3_scale3_topk1`
- `runs/gate12b_observer_relative_coarse_grained_closure_qwen2_5_0_5b_archive_motif_obs3_scale3_topk3`
- `runs/gate12b_observer_relative_coarse_grained_closure_qwen2_5_0_5b_archive_motif_obs3_scale3_topk5`
- `runs/gate12b_observer_relative_coarse_grained_closure_qwen2_5_3b_instruct_archive_motif_obs3_scale3_topk1`
- `runs/gate12b_observer_relative_coarse_grained_closure_qwen2_5_3b_instruct_archive_motif_obs3_scale3_topk3`
- `runs/gate12b_observer_relative_coarse_grained_closure_qwen2_5_3b_instruct_archive_motif_obs3_scale3_topk5`
- `runs/gate12b_observer_relative_coarse_grained_closure_llama3_2_3b_instruct_archive_motif_obs3_scale3_topk1`
- `runs/gate12b_observer_relative_coarse_grained_closure_llama3_2_3b_instruct_archive_motif_obs3_scale3_topk3`
- `runs/gate12b_observer_relative_coarse_grained_closure_llama3_2_3b_instruct_archive_motif_obs3_scale3_topk5`
- `runs/gate12b_observer_relative_coarse_grained_closure_qwen3_4b_archive_motif_obs3_scale3_topk1`
- `runs/gate12b_observer_relative_coarse_grained_closure_qwen3_4b_archive_motif_obs3_scale3_topk3`
- `runs/gate12b_observer_relative_coarse_grained_closure_qwen3_4b_archive_motif_obs3_scale3_topk5`

All runs are CPU-only read-only secondary audits over existing Gate12A archive
artifacts.

## 4. Results

### top_k = 1

| model | total | high | flat | high signature | flat signature | observer support | scale support |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| `Qwen/Qwen2.5-0.5B` | 3 | 1 | 2 | `residual_chord=3` / 1 | `residual_chord=1\|trusted_tree=2` / 2 | `3:2, 4:1` | `3:3` |
| `Qwen/Qwen2.5-3B-Instruct` | 3 | 1 | 2 | `residual_chord=3` / 1 | `residual_chord=1\|trusted_tree=2` / 2 | `3:2, 4:1` | `3:3` |
| `meta-llama/Llama-3.2-3B-Instruct` | 3 | 1 | 2 | `residual_chord=3` / 1 | `residual_chord=1\|trusted_tree=2` / 2 | `3:2, 4:1` | `3:3` |
| `Qwen/Qwen3-4B` | 3 | 1 | 2 | `residual_chord=3` / 1 | `residual_chord=1\|trusted_tree=2` / 2 | `3:2, 4:1` | `3:3` |

### top_k = 3

| model | total | high | flat | high signature | flat signature | observer support | scale support |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| `Qwen/Qwen2.5-0.5B` | 9 | 3 | 6 | `residual_chord=3` / 3 | `residual_chord=1\|trusted_tree=2` / 6 | `3:6, 4:3` | `3:9` |
| `Qwen/Qwen2.5-3B-Instruct` | 6 | 3 | 3 | `residual_chord=3` / 3 | `residual_chord=1\|trusted_tree=2` / 3 | `3:3, 4:3` | `3:6` |
| `meta-llama/Llama-3.2-3B-Instruct` | 9 | 3 | 6 | `residual_chord=3` / 3 | `residual_chord=1\|trusted_tree=2` / 6 | `3:6, 4:3` | `3:9` |
| `Qwen/Qwen3-4B` | 9 | 3 | 6 | `residual_chord=3` / 3 | `residual_chord=1\|trusted_tree=2` / 6 | `3:6, 4:3` | `3:9` |

### top_k = 5

| model | total | high | flat | high signature | flat signature | observer support | scale support |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| `Qwen/Qwen2.5-0.5B` | 15 | 5 | 10 | `residual_chord=3` / 5 | `residual_chord=1\|trusted_tree=2` / 10 | `3:10, 4:5` | `3:15` |
| `Qwen/Qwen2.5-3B-Instruct` | 11 | 5 | 6 | `residual_chord=3` / 5 | `residual_chord=1\|trusted_tree=2` / 6 | `3:6, 4:5` | `3:11` |
| `meta-llama/Llama-3.2-3B-Instruct` | 15 | 5 | 10 | `residual_chord=3` / 5 | `residual_chord=1\|trusted_tree=2` / 10 | `3:10, 4:5` | `3:15` |
| `Qwen/Qwen3-4B` | 15 | 5 | 10 | `residual_chord=3` / 5 | `residual_chord=1\|trusted_tree=2` / 10 | `3:10, 4:5` | `3:15` |

## 5. Gauge Boundary

All twelve motif observer-scope expansion runs had:

- `gauge_unstable_check_count = 0`
- `builder_script_sha256 = 1ac7821c7fcef4e15aa13debc2c9944f753d57b52a4d6da3f04b405dbd4af97d`
- `checksums.json` recomputation mismatches: `0`

The motif observer expansion does not weaken the existing bounded
basis-preserving local reparameterization check for these archive runs.

## 6. Reading Boundary

This memo earns the bounded sentence:

- with artifact-native ordered cycle-motif observers, the archive
  relation-signature flip survives `min_observer_support = 3` and
  `min_scale_support = 3` across the current dense-transformer archive set

It also restores a weak flat-side excess at this observer setting:

- `top_k = 1`: all four models are `flat 2 / high 1`
- `top_k = 3`: three models are `flat 6 / high 3`, while
  `Qwen/Qwen2.5-3B-Instruct` is balanced at `flat 3 / high 3`
- `top_k = 5`: all four models keep flat above high, with
  `Qwen/Qwen2.5-3B-Instruct` weaker at `flat 6 / high 5`

It does not earn:

- that `cycle_motif_expansion_v1` should replace `core_v1` as the default
- that observer-scope expansion proves a model-family law
- that candidate dominance is as strong as the baseline `220` run
- that sidecar architecture rows can be mixed into the dense-transformer claim
- that any candidate row is a failure label

## 7. Short Sentence

Under `cycle_motif_expansion_v1`, the archive relation-signature flip survives
the strict `min_observer_support = 3` and `min_scale_support = 3` setting:
`residual_chord=3` remains the high-tension side and
`residual_chord=1|trusted_tree=2` remains the flat side across all four current
dense-transformer archive surfaces for `top_k = 1 / 3 / 5`.
