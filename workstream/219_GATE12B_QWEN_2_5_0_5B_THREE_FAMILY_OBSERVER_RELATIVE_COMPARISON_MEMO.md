# Gate12B Qwen2.5-0.5B Three-Family Observer-Relative Comparison Memo

Status: first three-family Gate12B comparison memo draft
Role: bounded read-only secondary audit comparison over the current `Qwen/Qwen2.5-0.5B` Gate12A `transcript_v1 / briefing_v1 / archive_v1` family set, not a checkpoint revision, not a release claim, not a Gate12A schema change, and not an invariant-law promotion
Date: 2026-05-06

This memo proceeds from:

- `195_GATE11_ADMISSIBILITY_MEMBRANE_FREEZE.md`
- `196_GATE12_DISCRETE_CONNECTION_CONSTITUTION.md`
- `197_GATE12A_DISCRETE_CONNECTION_IMPLEMENTATION_CONTRACT.md`
- `198_GATE12A_DISCRETE_CONNECTION_AUDIT.md`
- `215_GATE12A_QWEN_2_5_0_5B_FIXED_FAMILY_SET_REPLICATION_MEMO.md`
- `217_GATE12B_OBSERVER_RELATIVE_COARSE_GRAINED_CLOSURE_OPENING_MEMO.md`
- `218_GATE12B_FIRST_OBSERVER_RELATIVE_COARSE_GRAINED_CLOSURE_SMOKE_MEMO.md`

## 0. Scope

This memo records the first bounded Gate12B comparison over three rendering
families for one fixed model line:

- model: `Qwen/Qwen2.5-0.5B`
- families: `transcript_v1 / briefing_v1 / archive_v1`
- secondary audit: `gate12b_observer_relative_coarse_grained_closure_v1`
- primitive: `observer_x_scale_x_admissible_gauge_transform_v1`
- gauge boundary: `basis_preserving_local_reparameterization_v1`

It does:

- run the same Gate12B scanner over the three current Gate12A discrete-connection artifacts
- compare invariant candidate count and candidate mix across the three families
- keep Gate12A artifacts read-only
- preserve the same bounded gauge transform used by the first Gate12B smoke

It does not:

- claim family-wide invariant law
- convert residual bands into correctness labels
- revise the Gate12A family-set memo
- create a checkpoint or release claim
- modify Gate12A contracts or schemas

## 1. Evidence Base

Source Gate12A runs:

- `runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_transcript_128r_gate9k/`
- `runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_briefing_200r_gate9k/`
- `runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_archive_128r_gate9k/`

Gate12B output runs:

- `runs/gate12b_observer_relative_coarse_grained_closure_qwen2_5_0_5b_transcript_family_compare/`
- `runs/gate12b_observer_relative_coarse_grained_closure_qwen2_5_0_5b_briefing_family_compare/`
- `runs/gate12b_observer_relative_coarse_grained_closure_qwen2_5_0_5b_archive_family_compare/`

All three Gate12B runs used:

- `top_k = 3`
- `flat_quantile = 0.25`
- `high_quantile = 0.75`
- `min_observer_support = 2`
- `min_scale_support = 2`
- `tau_gauge_residual_delta = 1e-08`
- `builder_script_sha256 = 550969db1577165b809e41c7fd04b00bb025f1e68dcc41e4635349f11b675126`

The commands are recorded in
`docs/gate12b_observer_relative_coarse_grained_closure_runbook.md`.

## 2. Run Matrix

| family | defined triangles | flat cut | high cut | flat | tense | high | matrix rows | invariant candidates | gauge-stable candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `transcript_v1` | 320 | 0.128943 | 0.696967 | 80 | 160 | 80 | 982 | 108 | 108 |
| `briefing_v1` | 500 | 0.100421 | 0.449925 | 125 | 250 | 125 | 1522 | 138 | 138 |
| `archive_v1` | 320 | 0.132193 | 2.000002 | 80 | 160 | 80 | 982 | 92 | 92 |

Gauge stability remained bounded in all three runs:

| family | gauge checks | stable checks | unstable checks | max residual delta abs |
| --- | ---: | ---: | ---: | ---: |
| `transcript_v1` | 3840 | 3840 | 0 | 4.440892098500626e-16 |
| `briefing_v1` | 6000 | 6000 | 0 | 4.440892098500626e-16 |
| `archive_v1` | 3840 | 3840 | 0 | 4.440892098500626e-16 |

## 3. Candidate Mix

| family | high-tension candidates | flat candidates | total candidates |
| --- | ---: | ---: | ---: |
| `transcript_v1` | 75 | 33 | 108 |
| `briefing_v1` | 112 | 26 | 138 |
| `archive_v1` | 21 | 71 | 92 |

The first-order shape is:

- `transcript_v1` and `briefing_v1` are high-tension-candidate dominant
- `briefing_v1` has the largest absolute candidate surface because its source triangle surface is larger
- `archive_v1` flips the mix and becomes flat-candidate dominant

## 4. Relation Signature Alignment

Candidate rows concentrate into two relation-kind signatures:

| family | dominant high-tension signature | count | dominant flat signature | count |
| --- | --- | ---: | --- | ---: |
| `transcript_v1` | <code>residual_chord=1&#124;trusted_tree=2</code> | 72 | `residual_chord=3` | 30 |
| `briefing_v1` | <code>residual_chord=1&#124;trusted_tree=2</code> | 109 | `residual_chord=3` | 23 |
| `archive_v1` | `residual_chord=3` | 18 | <code>residual_chord=1&#124;trusted_tree=2</code> | 68 |

This is the important Gate12B comparison signal in this first pass:

- transcript and briefing align `residual_chord=1|trusted_tree=2` with high-tension invariant candidates
- archive aligns that same mixed trusted-tree/residual-chord signature with flat invariant candidates
- archive instead places most high-tension candidates on the all-residual-chord signature

So the invariant candidate mechanism did not merely reproduce one global
signature. It preserved the same bounded observer/scale/gauge rules while
showing a family-conditioned signature-band alignment shift.

## 5. Reading Boundary

The honest reading is narrow:

- Gate12B found gauge-stable candidate rows in all three families
- the candidate count and candidate mix vary by family
- the largest qualitative shift is the archive relation-signature alignment flip
- this is a secondary audit over existing Gate12A artifacts, not a new Gate12A observable
- invariant candidates remain candidates, not laws

The next useful move is to repeat the same three-family Gate12B comparison on
the other current dense-transformer lines and ask whether the archive flip is
specific to `Qwen/Qwen2.5-0.5B`, shared across the lower-bound Qwen line, or a
broader rendering-family effect.

## 6. Short Sentence

Gate12B preserves bounded gauge stability across the current `Qwen/Qwen2.5-0.5B`
three-family set, while the invariant candidate surface is family-conditioned:
`transcript_v1` and `briefing_v1` are high-tension-candidate dominant, whereas
`archive_v1` flips to flat-candidate dominance and reverses the main
relation-signature-to-band alignment.
