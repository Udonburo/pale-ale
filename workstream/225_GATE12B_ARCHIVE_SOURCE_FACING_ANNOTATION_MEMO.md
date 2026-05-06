# Gate12B Archive Source-Facing Annotation Memo

Status: source-facing annotation memo draft
Role: bounded Gate12B archive source-facing annotation over the 224 inspection queue, not a checkpoint revision, not a release claim, not an invariant-law promotion, not an answer-quality label, and not a Gate12A/Gate12B schema change
Date: 2026-05-06

This memo proceeds from:

- `217_GATE12B_OBSERVER_RELATIVE_COARSE_GRAINED_CLOSURE_OPENING_MEMO.md`
- `218_GATE12B_FIRST_OBSERVER_RELATIVE_COARSE_GRAINED_CLOSURE_SMOKE_MEMO.md`
- `219_GATE12B_QWEN_2_5_0_5B_THREE_FAMILY_OBSERVER_RELATIVE_COMPARISON_MEMO.md`
- `220_GATE12B_DENSE_TRANSFORMER_FAMILY_EFFECT_EXPANSION_MEMO.md`
- `221_GATE12B_ARCHIVE_STRICT_SUPPORT_SENSITIVITY_MEMO.md`
- `222_GATE12B_ARCHIVE_OBSERVER_SCOPE_EXPANSION_SENSITIVITY_MEMO.md`
- `223_GATE12B_MOTIF_OBSERVER_SPECIFICITY_CHECK.md`
- `224_GATE12B_ARCHIVE_CANDIDATE_SOURCE_INSPECTION_QUEUE.md`

## 0. Scope

This memo fixes a source-facing annotation layer over the `224` archive
inspection queue.

The question is:

```text
Given the 16 queued archive rows, what source-facing direction does each row
show: support-following, conflict-following, non-gluing, or ambiguous?
```

This pass does not:

- run model inference
- add a new observer mode
- change Gate12A or Gate12B artifacts
- turn a candidate row into a model-quality label
- treat exact anchor matching as a semantic evaluator

The annotation is intentionally source-facing. It only records how the queued
answer surface reads relative to the support/conflict source anchors available
in the Gate12A text-surface artifact.

## 1. Added Annotation Helper

This pass adds:

- `tools/annotate_gate12b_source_inspection_queue.py`

The helper reads:

- `gate12b_source_inspection_queue.jsonl`

and emits:

- `gate12b_source_annotations.jsonl`
- `gate12b_source_annotations.csv`
- `gate12b_source_annotation_summary.json`
- `gate12b_source_annotation_summary.csv`
- `gate12b_source_annotation.md`
- `manifest.json`
- `checksums.json`

The helper supports two modes:

- derived annotation from queue fields
- supplied annotation validation

The allowed tag vocabulary is:

- `support-following`
- `conflict-following`
- `non-gluing`
- `ambiguous`

The derived mode used here is:

- `exact_anchor_and_non_gluing_phrase_v1`

Rules:

- exact conflict-anchor text in answer -> `conflict-following`
- exact support-anchor text in answer -> `support-following`
- no-gluing / not-warranted phrasing -> `non-gluing`
- otherwise -> `ambiguous`

These are annotation rules for the source-facing queue only. They are not a
general answer evaluator.

## 2. Run

The annotation artifact is:

- `runs/gate12b_archive_source_facing_annotation_motif_topk3/`

Input queue:

- `runs/gate12b_archive_candidate_source_inspection_queue_motif_topk3/gate12b_source_inspection_queue.jsonl`

Status:

| field | value |
| --- | ---: |
| annotation rows | 16 |
| summary rows | 3 |
| flat rows | 8 |
| high-tension rows | 8 |
| support-following | 6 |
| conflict-following | 8 |
| non-gluing | 2 |
| ambiguous | 0 |
| high-tension conflict-following | 8 |
| flat support-or-non-gluing | 8 |
| checksum mismatches | 0 |
| builder script sha256 | `b115b986e2952b92517d56f87e6a92a269a101c89291bf295c0e42585a02426d` |

## 3. Annotation Summary

| candidate side | relation signature | source-facing tag | count |
| --- | --- | --- | ---: |
| `flat` | `residual_chord=1\|trusted_tree=2` | `support-following` | 6 |
| `flat` | `residual_chord=1\|trusted_tree=2` | `non-gluing` | 2 |
| `high_tension` | `residual_chord=3` | `conflict-following` | 8 |

This is the clean source-facing correspondence:

```text
high R -> conflict-following
flat M -> support-following or non-gluing
```

## 4. Reading Boundary

This memo earns a bounded source-facing sentence:

- over the selected archive source-inspection queue, high-side
  `residual_chord=3` candidates are source-facing `conflict-following` in
  `8/8` rows, while flat-side `residual_chord=1|trusted_tree=2` candidates are
  `support-following` in `6/8` rows and `non-gluing` in `2/8` rows

It does not earn:

- a full semantic phenotype claim
- a model-quality claim
- a universal archive law
- that exact anchor matching is sufficient for all source inspection
- that source-facing annotation should become an automated evaluator

## 5. Why This Matters

`220` through `223` established a structural archive flip under Gate12B.
`224` returned the candidate rows to source text.

This memo adds the missing intermediate layer:

```text
Gate12B archive flip
-> source queue
-> source-facing annotation
```

The result is useful because it keeps the structural signal and the source read
separate. The source-facing annotation supports the structural read without
collapsing the Gate12B candidate into a quality score.

## 6. Next Step

The next useful step is not a new observer.

The clean options are:

- run the same queue + annotation path for `transcript_v1` and `briefing_v1`
- or widen archive source-facing annotation beyond the selected `per_band_limit = 2`

The first option tests family specificity. The second option tests archive
coverage.

## 7. Short Sentence

The Gate12B archive source-facing annotation fixes the 224 queue into a clean
source read: selected high-side `residual_chord=3` rows are
`conflict-following` in `8/8`, while selected flat-side
`residual_chord=1|trusted_tree=2` rows are `support-following` in `6/8` and
`non-gluing` in `2/8`.
