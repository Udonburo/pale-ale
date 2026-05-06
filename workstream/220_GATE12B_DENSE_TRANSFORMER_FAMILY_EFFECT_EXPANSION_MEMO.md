# Gate12B Dense-Transformer Family-Effect Expansion Memo

Status: first expanded Gate12B family-effect comparison memo draft
Role: bounded read-only Gate12B expansion across the current dense-transformer fixed family sets, not a checkpoint revision, not a release claim, not a Gate12A schema change, and not an invariant-law promotion
Date: 2026-05-06

This memo proceeds from:

- `195_GATE11_ADMISSIBILITY_MEMBRANE_FREEZE.md`
- `196_GATE12_DISCRETE_CONNECTION_CONSTITUTION.md`
- `197_GATE12A_DISCRETE_CONNECTION_IMPLEMENTATION_CONTRACT.md`
- `198_GATE12A_DISCRETE_CONNECTION_AUDIT.md`
- `210_GATE12A_QWEN_2_5_3B_INSTRUCT_FIXED_FAMILY_SET_REPLICATION_MEMO.md`
- `211_GATE12A_LLAMA_3_2_3B_INSTRUCT_FIXED_FAMILY_SET_REPLICATION_MEMO.md`
- `212_GATE12A_QWEN3_4B_FIXED_FAMILY_SET_REPLICATION_MEMO.md`
- `215_GATE12A_QWEN_2_5_0_5B_FIXED_FAMILY_SET_REPLICATION_MEMO.md`
- `217_GATE12B_OBSERVER_RELATIVE_COARSE_GRAINED_CLOSURE_OPENING_MEMO.md`
- `218_GATE12B_FIRST_OBSERVER_RELATIVE_COARSE_GRAINED_CLOSURE_SMOKE_MEMO.md`
- `219_GATE12B_QWEN_2_5_0_5B_THREE_FAMILY_OBSERVER_RELATIVE_COMPARISON_MEMO.md`

## 0. Scope

This memo expands the Gate12B comparison from one model line to the current
dense-transformer fixed family set:

- `Qwen/Qwen2.5-0.5B`
- `Qwen/Qwen2.5-3B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`
- `Qwen/Qwen3-4B`

Each model is compared across:

- `transcript_v1`
- `briefing_v1`
- `archive_v1`

It uses the same Gate12B secondary audit surface:

- `gate12b_observer_relative_coarse_grained_closure_v1`
- `observer_x_scale_x_admissible_gauge_transform_v1`
- `basis_preserving_local_reparameterization_v1`
- `top_k = 3`
- `flat_quantile = 0.25`
- `high_quantile = 0.75`

It does not:

- promote invariant candidates into a law
- convert residual bands into correctness labels
- revise Gate12A family-set memos
- create a checkpoint or release claim
- include the non-transformer sidecar line

## 1. Evidence Base

The comparison reads existing Gate12A discrete-connection artifacts and emits
Gate12B outputs under `runs/`.

New output runs added by this expansion:

- `runs/gate12b_observer_relative_coarse_grained_closure_qwen2_5_3b_instruct_transcript_family_compare/`
- `runs/gate12b_observer_relative_coarse_grained_closure_qwen2_5_3b_instruct_briefing_family_compare/`
- `runs/gate12b_observer_relative_coarse_grained_closure_qwen2_5_3b_instruct_archive_family_compare/`
- `runs/gate12b_observer_relative_coarse_grained_closure_llama3_2_3b_instruct_transcript_family_compare/`
- `runs/gate12b_observer_relative_coarse_grained_closure_llama3_2_3b_instruct_briefing_family_compare/`
- `runs/gate12b_observer_relative_coarse_grained_closure_llama3_2_3b_instruct_archive_family_compare/`
- `runs/gate12b_observer_relative_coarse_grained_closure_qwen3_4b_transcript_family_compare/`
- `runs/gate12b_observer_relative_coarse_grained_closure_qwen3_4b_briefing_family_compare/`
- `runs/gate12b_observer_relative_coarse_grained_closure_qwen3_4b_archive_family_compare/`

The `Qwen/Qwen2.5-0.5B` comparison runs remain the three outputs recorded in
`219`.

All twelve family-compare manifests record:

- `builder_script_sha256 = 550969db1577165b809e41c7fd04b00bb025f1e68dcc41e4635349f11b675126`
- no `checksums.json` mismatches

## 2. Candidate Dominance Matrix

This table records candidate-band dominance, not correctness.

| model | transcript | briefing | archive |
| --- | --- | --- | --- |
| `Qwen/Qwen2.5-0.5B` | high, 75/33 | high, 112/26 | flat, 21/71 |
| `Qwen/Qwen2.5-3B-Instruct` | high, 77/29 | high, 118/57 | flat, 35/73 |
| `meta-llama/Llama-3.2-3B-Instruct` | high, 71/23 | flat, 37/124 | flat, 37/77 |
| `Qwen/Qwen3-4B` | flat, 23/75 | high, 112/43 | flat, 37/69 |

Cell notation is:

- dominant band
- `high_candidate_count / flat_candidate_count`

The first expanded result is:

- all four `archive_v1` surfaces are flat-candidate dominant
- `transcript_v1` is high-dominant for both Qwen2.5 lines and Llama3.2, but flat-dominant for Qwen3
- `briefing_v1` is high-dominant for the Qwen lines, but flat-dominant for Llama3.2

So the archive behavior looks like a rendering-family effect in this bounded
set. Transcript and briefing remain model-conditioned.

## 3. Relation Signature Alignment

For compactness, this memo names the two dominant relation signatures:

- `M = residual_chord=1|trusted_tree=2`
- `R = residual_chord=3`

| model | family | dominant high signature | dominant flat signature |
| --- | --- | --- | --- |
| `Qwen/Qwen2.5-0.5B` | `transcript_v1` | M, 72 | R, 30 |
| `Qwen/Qwen2.5-0.5B` | `briefing_v1` | M, 109 | R, 23 |
| `Qwen/Qwen2.5-0.5B` | `archive_v1` | R, 18 | M, 68 |
| `Qwen/Qwen2.5-3B-Instruct` | `transcript_v1` | M, 74 | R, 26 |
| `Qwen/Qwen2.5-3B-Instruct` | `briefing_v1` | M, 115 | R, 54 |
| `Qwen/Qwen2.5-3B-Instruct` | `archive_v1` | R, 32 | M, 70 |
| `meta-llama/Llama-3.2-3B-Instruct` | `transcript_v1` | M, 68 | R, 20 |
| `meta-llama/Llama-3.2-3B-Instruct` | `briefing_v1` | R, 34 | M, 121 |
| `meta-llama/Llama-3.2-3B-Instruct` | `archive_v1` | R, 34 | M, 74 |
| `Qwen/Qwen3-4B` | `transcript_v1` | R, 20 | M, 72 |
| `Qwen/Qwen3-4B` | `briefing_v1` | M, 109 | R, 40 |
| `Qwen/Qwen3-4B` | `archive_v1` | R, 34 | M, 66 |

The archive row is the bounded repeated signal:

- every `archive_v1` run maps high-tension candidates mostly to `R`
- every `archive_v1` run maps flat candidates mostly to `M`

That is the same alignment flip first observed in `219`, now repeated across
all four current dense-transformer lines.

## 4. Gauge Boundary

The bounded gauge check mostly stays quiet:

- eleven of twelve family-compare runs have `gauge_unstable_check_count = 0`
- all twelve manifests match the current runner hash
- all twelve `checksums.json` files match their emitted artifacts

One narrow exception appears:

- run: `runs/gate12b_observer_relative_coarse_grained_closure_qwen3_4b_briefing_family_compare/`
- `gauge_total_check_count = 6000`
- `gauge_stable_check_count = 5988`
- `gauge_unstable_check_count = 12`
- `max_residual_delta_abs = 1.6653345369377348e-16`
- unstable cycle: `triangle:000358`
- band movement: `tense -> flat`

The residual delta is far below `tau_gauge_residual_delta = 1e-08`.
The instability is a residual-band boundary effect around the flat cut, not a
large metric residual movement. `triangle:000358` is not an invariant candidate
row in this run, and the run still emits:

- `invariant_signature_candidate_count = 155`
- `gauge_variant_signature_candidate_count = 155`

So this exception should be carried as a threshold-boundary caveat, not as a
rejection of the expanded candidate comparison.

## 5. Reading Boundary

This memo earns the bounded sentence:

- Gate12B's archive-family candidate surface repeats across the current dense-transformer set as flat-dominant with the same relation-signature alignment flip

It does not earn:

- that archive behavior is universal
- that transcript or briefing behavior is model-invariant
- that invariant candidates are physical invariants
- that residual bands are correctness classes
- that Gate12B changes the Gate12A published checkpoint boundary

The next useful move is a focused archive-family read:

- ask why `archive_v1` consistently maps `M` to flat candidates and `R` to high candidates
- inspect whether the same archive flip appears under stricter candidate support
- decide whether Gate12B should add a dedicated family-effect summary artifact before any release-bound packaging

## 6. Short Sentence

Across the current dense-transformer Gate12B expansion, `archive_v1` is the
bounded repeated family-conditioned signal: all four archive runs are flat-candidate
dominant and share the same relation-signature alignment flip, while transcript
and briefing remain model-conditioned rather than globally invariant.
