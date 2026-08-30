# Gate12B Paper Evidence Package Manifest

Status: release-facing compact evidence manifest
Role: bounded manifest for the archived Gate12B paper package; records local evidence paths, hashes, checksum expectations, and paper-table mappings without including generated `runs/` artifacts, changing experiments, or expanding claims
Date: 2026-05-06

This release-facing manifest is derived from the following workstream evidence
memos:

- `217_GATE12B_OBSERVER_RELATIVE_COARSE_GRAINED_CLOSURE_OPENING_MEMO.md`
- `218_GATE12B_FIRST_OBSERVER_RELATIVE_COARSE_GRAINED_CLOSURE_SMOKE_MEMO.md`
- `219_GATE12B_QWEN_2_5_0_5B_THREE_FAMILY_OBSERVER_RELATIVE_COMPARISON_MEMO.md`
- `220_GATE12B_DENSE_TRANSFORMER_FAMILY_EFFECT_EXPANSION_MEMO.md`
- `221_GATE12B_ARCHIVE_STRICT_SUPPORT_SENSITIVITY_MEMO.md`
- `222_GATE12B_ARCHIVE_OBSERVER_SCOPE_EXPANSION_SENSITIVITY_MEMO.md`
- `223_GATE12B_MOTIF_OBSERVER_SPECIFICITY_CHECK.md`
- `224_GATE12B_ARCHIVE_CANDIDATE_SOURCE_INSPECTION_QUEUE.md`
- `225_GATE12B_ARCHIVE_SOURCE_FACING_ANNOTATION_MEMO.md`
- `226_GATE12B_NONARCHIVE_SOURCE_FACING_ANNOTATION_SENSITIVITY.md`
- `227_GATE12B_OBSERVER_RELATIVE_CLOSURE_CLOSEOUT_MEMO.md`
- `228_GATE12B_PAPER_OUTLINE_AND_CLAIM_BOUNDARY.md`

## 0. Scope

This manifest records the compact evidence package boundary for the Gate12B
paper line.

The archived paper package does not include generated `runs/` artifacts. It
lists the local artifact directories, hashes, checksum expectations, and
paper-table mappings used to verify the manuscript tables and evidence chain.

The package purpose is narrow:

```text
support the bounded Gate12B paper claim without reopening experiments,
observer design, or theory vocabulary
```

## 1. Package Boundary

Package manifest id:

```text
gate12b_paper_evidence_package_manifest_v1
```

Packaged record status:

```text
release_manifest_with_local_evidence_record_no_runs_payload
```

This compact release package includes the manuscript, TeX source, this
release-facing evidence manifest, and package checksums. It intentionally does
not turn `runs/` into a tracked dump.

Do not include:

- unrelated Gate8/Gate9/Gate12A exploratory runs
- local scratch notes
- temporary fixtures
- unselected sidecar architecture runs
- generated artifacts without a table or claim mapping
- new model inference outputs

## 2. Paper Claim Fragments

| claim fragment | required evidence set |
| --- | --- |
| Gate12B is a read-only secondary audit over existing Gate12A artifacts | Gate12B runner outputs plus workstream `217`, `218`, `227`, `228` |
| Archive high-side `residual_chord=3` / flat-side `residual_chord=1\|trusted_tree=2` repeats across four dense-transformer model lines | motif specificity summary plus 36 motif Gate12B runs |
| The archive direction is observed under the representative motif path and scale-focused support path; core-observer boundary path is a candidate-vanish boundary | archive motif top-k runs plus archive boundary runs |
| The motif observer sensitivity does not force transcript/briefing into archive direction | motif specificity summary plus 36 motif Gate12B runs |
| Archive source-facing rows align high R with conflict and flat M with support-or-non-gluing | archive source inspection queue and archive source-facing annotation artifacts |
| Transcript/briefing do not reproduce the same clean archive alignment | transcript/briefing source queues and source-facing annotations |
| The result remains bounded and non-universal | workstream `227` and `228` |

## 3. Required Evidence Sets

| set id | artifact class | local directories | paper role | checksum status |
| --- | --- | ---: | --- | --- |
| `motif_specificity_summary` | Gate12B run summary | 1 | primary Table 1 / Table 2 source | summary artifact has file hashes below; input rows report checksum `ok` |
| `motif_gate12b_runs` | Gate12B run directories | 36 | regenerate motif specificity summary and family comparison | 36/36 `checksums.json` present; mismatch count `0` |
| `source_queue_annotation` | source queue and annotation artifacts | 6 | Table 3 source-facing annotation evidence | 6/6 `checksums.json` present; mismatch count `0` |
| `archive_core_boundary_runs` | Gate12B archive boundary runs | 24 | support `221` / `222` boundary discussion | 24/24 `checksums.json` present; mismatch count `0` |
| `upstream_rebuild_inputs` | Gate12A discrete connection and text-surface inputs | local upstream dependency set | optional rebuild provenance | not included in this compact release package |

The minimal paper package can start from the first three sets. The
`archive_core_boundary_runs` set should be included if the paper table or
appendix explicitly discusses the default core observer boundary from `221` and
`222`.

## 4. Motif Specificity Summary

Primary summary artifact:

- `runs/gate12b_summary_motif_specificity_dense_transformer/`

Files:

| file | sha256 |
| --- | --- |
| `manifest.json` | `df2b43d52d6f3f3a4d980e3f5daf2a818d3cbed71a2026abbff61951f549ab9f` |
| `gate12b_run_summary.csv` | `a96b6780895d527ce949664116dc59f5df222a91cf5b8c054ec9a8ce3abad1ca` |
| `gate12b_run_summary.json` | `61f813921c8c96132661373ce5106ebf0bdace040019c7a92e696288ab1d36a3` |

Manifest status:

| field | value |
| --- | --- |
| schema | `gate12b_run_summary_v1` |
| run count | 36 |
| current Gate12B runner sha256 | `1ac7821c7fcef4e15aa13debc2c9944f753d57b52a4d6da3f04b405dbd4af97d` |

The summary artifact does not currently include a `checksums.json` file. It is
therefore identified by direct file hashes above and by the checksum status of
the 36 input runs.

## 5. Motif Gate12B Run Set

This set contains the 36 representative motif-path runs used by `223` and
`228`.

Grid:

| axis | values |
| --- | --- |
| families | `archive_v1`, `transcript_v1`, `briefing_v1` |
| models | `qwen2_5_0_5b`, `qwen2_5_3b_instruct`, `llama3_2_3b_instruct`, `qwen3_4b` |
| `top_k` | `1`, `3`, `5` |
| observer mode set | `cycle_motif_expansion_v1` |
| minimum observer support | `3` |
| minimum scale support | `3` |

Directory template:

```text
runs/gate12b_observer_relative_coarse_grained_closure_{model}_{family}_motif_obs3_scale3_topk{top_k}/
```

Required files per run:

- `manifest.json`
- `observer_scale_closure_matrix.csv`
- `observer_scale_closure_matrix.json`
- `invariant_signature_candidates.jsonl`
- `gauge_stability_matrix.csv`
- `gauge_stability_summary.json`
- `gauge_variant_signature_candidates.jsonl`
- `gate12b_observer_relative_coarse_grained_closure.md`
- `checksums.json`

Integrity status:

| field | value |
| --- | --- |
| run directories | 36 |
| `checksums.json` present | 36 / 36 |
| recomputed checksum mismatches | 0 |
| builder script sha256 | `1ac7821c7fcef4e15aa13debc2c9944f753d57b52a4d6da3f04b405dbd4af97d` in 36 / 36 |

This set supports:

- archive relation-signature repetition
- transcript/briefing non-archive sensitivity
- motif observer specificity
- Qwen3-4B briefing threshold-boundary caveat

## 6. Reparameterization Availability and Outcome Boundary

The paper reports reparameterization availability as a boundary check, not as
independent support for the archive-family result. Counts are recorded from the
`gauge_stability_summary.json` files in the motif run set and archive boundary
set.

| evidence group | source summary artifacts | nontrivial `basis_coordinate_reversal_v1` checks | registry-only identity fallback cases | failed check rows | treatment |
| --- | --- | ---: | ---: | ---: | --- |
| motif `archive_v1` | 12 `runs/gate12b_observer_relative_coarse_grained_closure_*_archive_motif_obs3_scale3_topk*/gauge_stability_summary.json` files | 12 / 12 runs; 67,584 check rows | 0 / 12 runs | 0 | zero failed rows; not independent support |
| motif `transcript_v1` | 12 `runs/gate12b_observer_relative_coarse_grained_closure_*_transcript_motif_obs3_scale3_topk*/gauge_stability_summary.json` files | 12 / 12 runs; 67,584 check rows | 0 / 12 runs | 0 | non-archive sensitivity boundary |
| motif `briefing_v1` | 12 `runs/gate12b_observer_relative_coarse_grained_closure_*_briefing_motif_obs3_scale3_topk*/gauge_stability_summary.json` files | 12 / 12 runs; 105,600 check rows | 0 / 12 runs | 48 | threshold-boundary rows; not candidate-level instability |
| archive boundary | 24 archive boundary `gauge_stability_summary.json` files listed in Section 8 | 24 / 24 runs; 92,160 check rows | 0 / 24 runs | 0 | boundary evidence for archive sensitivity paths |

Registry-only identity fallback cases are not counted as nontrivial
reparameterization evidence. Failed check rows are read from
`unstable_check_count` in the corresponding `gauge_stability_summary.json`
files.

The larger briefing denominator reflects the larger number of
reparameterization check rows materialized in the recorded Gate12B
`gauge_stability_summary.json` artifacts, not a different reparameterization
criterion.

## 7. Source Queue and Annotation Set

This set contains the selected source-facing queue and derived annotation
artifacts from `224`, `225`, and `226`.

| family | queue artifact | annotation artifact | rows | annotation status |
| --- | --- | --- | ---: | --- |
| `archive_v1` | `runs/gate12b_archive_candidate_source_inspection_queue_motif_topk3/` | `runs/gate12b_archive_source_facing_annotation_motif_topk3/` | 16 queue / 16 annotation | `conflict-following=8`, `support-following=6`, `non-gluing=2`, `ambiguous=0` |
| `transcript_v1` | `runs/gate12b_transcript_candidate_source_inspection_queue_motif_topk3/` | `runs/gate12b_transcript_source_facing_annotation_motif_topk3/` | 16 queue / 16 annotation | `conflict-following=6`, `support-following=10`, `non-gluing=0`, `ambiguous=0` |
| `briefing_v1` | `runs/gate12b_briefing_candidate_source_inspection_queue_motif_topk3/` | `runs/gate12b_briefing_source_facing_annotation_motif_topk3/` | 16 queue / 16 annotation | `conflict-following=4`, `support-following=12`, `non-gluing=0`, `ambiguous=0` |

Queue builder hashes:

| queue artifact | builder script sha256 |
| --- | --- |
| archive queue | `8b0ac133af46e08c60715e44556200cdecdf3d9d74ba09e95918f312e0f5bb5e` |
| transcript queue | `fb2f2bcd569c176b081207faabe765908d115dcfa4df318950ca6b7e6f567b01` |
| briefing queue | `fb2f2bcd569c176b081207faabe765908d115dcfa4df318950ca6b7e6f567b01` |

Annotation helper hashes:

| annotation artifact | builder script sha256 |
| --- | --- |
| archive annotation | `b115b986e2952b92517d56f87e6a92a269a101c89291bf295c0e42585a02426d` |
| transcript annotation | `ffe7128e7e96358308f2eef1e9ae12b9a2fa4eb66e6a34c06134ba04008aec81` |
| briefing annotation | `ffe7128e7e96358308f2eef1e9ae12b9a2fa4eb66e6a34c06134ba04008aec81` |

Integrity status:

| field | value |
| --- | --- |
| source queue / annotation directories | 6 |
| `checksums.json` present | 6 / 6 |
| recomputed checksum mismatches | 0 |
| queue source linkage | 12 / 12 cases have matching Gate12B/text-surface `source_gate12a_run_id` |

Source-facing annotation remains exact-anchor / phrase-rule derived
annotation. It is not a semantic evaluator and not an answer-quality label.

## 8. Archive Boundary Set

This set supports the boundary discussion from `221` and `222`. It should be
included in an appendix package if the paper discusses why the default core
observer setting is narrower than `cycle_motif_expansion_v1`.

Directory families:

```text
runs/gate12b_observer_relative_coarse_grained_closure_{model}_archive_strict_obs2_scale3_topk{1,3,5}/
runs/gate12b_observer_relative_coarse_grained_closure_{model}_archive_strict_obs3_scale2_topk3/
runs/gate12b_observer_relative_coarse_grained_closure_{model}_archive_strict_topk{1,5}/
```

Models:

- `qwen2_5_0_5b`
- `qwen2_5_3b_instruct`
- `llama3_2_3b_instruct`
- `qwen3_4b`

Integrity status:

| field | value |
| --- | --- |
| run directories | 24 |
| `checksums.json` present | 24 / 24 |
| recomputed checksum mismatches | 0 |
| builder script sha256 | `550969db1577165b809e41c7fd04b00bb025f1e68dcc41e4635349f11b675126` in 24 / 24 |

This is a boundary evidence set, not the main paper-result set.

## 9. Upstream Rebuild Inputs

If the evidence package needs rebuild provenance rather than table-level
evidence only, include the upstream Gate12A inputs referenced by the selected
Gate12B and source queue artifacts.

Gate12A discrete connection input template:

```text
runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_{model}_{family}_{row_count}_gate9k/
```

Gate12A triangle text-surface input template:

```text
runs/gate12a_triangle_text_surface_audit_recheck_from_gate12a_upstream_gate8cm_{model}_{family}_{row_count}_gate9k/
```

Selected models and families:

| family | row count | models |
| --- | --- | --- |
| `archive_v1` | `128r` | Qwen2.5-0.5B, Qwen2.5-3B-Instruct, Llama-3.2-3B-Instruct, Qwen3-4B |
| `transcript_v1` | `128r` | Qwen2.5-0.5B, Qwen2.5-3B-Instruct, Llama-3.2-3B-Instruct, Qwen3-4B |
| `briefing_v1` | `200r` | Qwen2.5-0.5B, Qwen2.5-3B-Instruct, Llama-3.2-3B-Instruct, Qwen3-4B |

These upstream inputs are not required if the package only preserves the
Gate12B summaries and source-facing derived annotation outputs. They are
required if a reviewer needs to rebuild the selected Gate12B queues from
Gate12A text-surface rows.

## 10. Paper Table Mapping

| paper item from `228` | evidence package set |
| --- | --- |
| Figure 1: Gate12A -> Gate12B -> source queue -> source-facing annotation pipeline | source queue and annotation set; upstream rebuild inputs for provenance |
| Table 1: Dense-transformer family comparison | motif specificity summary; motif Gate12B run set |
| Table 2: Scale-focused support path across `top_k=1/3/5` | motif Gate12B archive runs; archive boundary set if appendix includes default-core boundary |
| Table 3: Source-facing annotation summary | source queue and annotation set |
| Table 4: Non-claims and boundary conditions | workstream `227`, `228`, `229` |
| Table 5: Caveats | motif specificity summary; Qwen3-4B briefing motif runs; `223` and `227` text |
| Reparameterization availability table | motif Gate12B run set; archive boundary set; Section 6 of this manifest |

## 11. Future Full Evidence Bundle Procedure

If a future full evidence bundle includes generated run directories:

1. Start from a clean `main`.
2. Create a dedicated evidence-package branch or external archive, not a
   regular workstream memo branch.
3. Copy only the directories listed in this manifest.
4. Preserve each artifact directory's `manifest.json` and `checksums.json`.
5. Recompute checksums and record mismatches.
6. Include this manifest as the package manifest.
7. Do not add unrelated `runs/` directories.
8. Do not regenerate model outputs.

Recommended package name:

```text
gate12b_paper_evidence_package_v1
```

## 12. Reading Boundary

This manifest does not promote the Gate12B result beyond `228`.

It does not claim:

- a universal law
- a model-quality benchmark
- a correctness classifier
- a physical gauge invariant
- a checkpoint or release result
- a claim about model weights in general

It also does not introduce new observer modes or new source-facing annotation
semantics.

## 13. Source Lineage

This file is a release-facing derivative of:

```text
workstream/229_GATE12B_PAPER_EVIDENCE_PACKAGE_MANIFEST.md
```

It preserves the evidence mapping from the workstream manifest while removing
workstream handoff and local-only packaging language. The packaged manuscript
source is:

```text
gate12b-observer-relative-closure-signatures.tex
```

## 14. Short Sentence

The Gate12B paper evidence package should be a small, intentional local
artifact set: the motif specificity summary and runs, the selected
source-facing queues and annotations, and the archive boundary runs needed for
appendix sensitivity, all mapped directly to paper claims and tables without
committing unrelated `runs/` output.
