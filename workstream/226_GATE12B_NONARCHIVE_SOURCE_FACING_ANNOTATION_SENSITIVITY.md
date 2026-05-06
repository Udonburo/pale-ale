# Gate12B Non-Archive Source-Facing Annotation Sensitivity Memo

Status: non-archive source-facing annotation sensitivity memo draft
Role: bounded Gate12B transcript/briefing source-facing annotation sensitivity over existing Gate12A/Gate12B artifacts, not a checkpoint revision, not a release claim, not an invariant-law promotion, not an answer-quality label, and not a Gate12A/Gate12B schema change
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
- `225_GATE12B_ARCHIVE_SOURCE_FACING_ANNOTATION_MEMO.md`

## 0. Scope

This memo applies the same source inspection queue plus source-facing derived
annotation path from `224` and `225` to the non-archive families:

- `transcript_v1`
- `briefing_v1`

The question is:

```text
Does the source-facing archive alignment from 225 appear family-specific, or
does the annotation rule produce the same alignment for transcript_v1 and
briefing_v1?
```

The sensitivity uses the representative strict motif setting:

- `observer_mode_set = cycle_motif_expansion_v1`
- `top_k = 3`
- `min_observer_support = 3`
- `min_scale_support = 3`
- `per_band_limit = 2`

It covers:

- `Qwen/Qwen2.5-0.5B`
- `Qwen/Qwen2.5-3B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`
- `Qwen/Qwen3-4B`

This pass does not:

- run model inference
- add a new observer mode
- change Gate12A or Gate12B source artifacts
- change Gate12A or Gate12B schemas, thresholds, or classifications
- turn source-facing tags into answer-quality labels
- treat derived exact-anchor matching as human semantic annotation

## 1. Runs

The non-archive source inspection queue artifacts are:

- `runs/gate12b_transcript_candidate_source_inspection_queue_motif_topk3/`
- `runs/gate12b_briefing_candidate_source_inspection_queue_motif_topk3/`

The non-archive source-facing annotation artifacts are:

- `runs/gate12b_transcript_source_facing_annotation_motif_topk3/`
- `runs/gate12b_briefing_source_facing_annotation_motif_topk3/`

The queue builder validated that every Gate12B run and text-surface audit
shared the same `source_gate12a_run_id`. The output directories were outside
the input artifact directories.

Builder hashes for this pass:

| helper | builder script sha256 |
| --- | --- |
| queue builder | `fb2f2bcd569c176b081207faabe765908d115dcfa4df318950ca6b7e6f567b01` |
| source-facing annotation helper | `ffe7128e7e96358308f2eef1e9ae12b9a2fa4eb66e6a34c06134ba04008aec81` |

## 2. Transcript Queue and Annotation Status

| family | cases | queue rows | flat rows | high-tension rows | support-following | conflict-following | non-gluing | ambiguous | checksum mismatches | builder script sha256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `transcript_v1` | 4 | 16 | 8 | 8 | 10 | 6 | 0 | 0 | 0 queue, 0 annotation | `ffe7128e7e96358308f2eef1e9ae12b9a2fa4eb66e6a34c06134ba04008aec81` |

Transcript source-facing summary:

| candidate side | relation signature | source-facing tag | count |
| --- | --- | --- | ---: |
| `flat` | `residual_chord=3` | `support-following` | 6 |
| `flat` | `residual_chord=1\|trusted_tree=2` | `support-following` | 2 |
| `high_tension` | `residual_chord=1\|trusted_tree=2` | `conflict-following` | 4 |
| `high_tension` | `residual_chord=1\|trusted_tree=2` | `support-following` | 2 |
| `high_tension` | `residual_chord=3` | `conflict-following` | 2 |

Transcript keeps a source-facing high/flat split in aggregate:

- high-tension rows are `conflict-following` in `6/8`
- flat rows are `support-following` in `8/8`

But the dominant relation signatures mostly reverse the archive direction:

- high side: `residual_chord=1|trusted_tree=2` in `6/8`
- flat side: `residual_chord=3` in `6/8`

The Qwen3-4B transcript case is the archive-direction exception:

- high side: `residual_chord=3`, `conflict-following`
- flat side: `residual_chord=1|trusted_tree=2`, `support-following`

## 3. Briefing Queue and Annotation Status

| family | cases | queue rows | flat rows | high-tension rows | support-following | conflict-following | non-gluing | ambiguous | checksum mismatches | builder script sha256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `briefing_v1` | 4 | 16 | 8 | 8 | 12 | 4 | 0 | 0 | 0 queue, 0 annotation | `ffe7128e7e96358308f2eef1e9ae12b9a2fa4eb66e6a34c06134ba04008aec81` |

Briefing source-facing summary:

| candidate side | relation signature | source-facing tag | count |
| --- | --- | --- | ---: |
| `flat` | `residual_chord=1\|trusted_tree=2` | `support-following` | 2 |
| `flat` | `residual_chord=3` | `support-following` | 4 |
| `flat` | `residual_chord=3` | `conflict-following` | 2 |
| `high_tension` | `residual_chord=1\|trusted_tree=2` | `support-following` | 6 |
| `high_tension` | `residual_chord=3` | `conflict-following` | 2 |

Briefing is not archive-like under the source-facing derived annotation:

- high-tension rows are `support-following` in `6/8`
- flat rows are `support-following` in `6/8` and `conflict-following` in `2/8`

The dominant relation signatures again mostly reverse the archive direction:

- high side: `residual_chord=1|trusted_tree=2` in `6/8`
- flat side: `residual_chord=3` in `6/8`

The model cases are mixed:

- Llama-3.2-3B-Instruct briefing is archive-direction by relation signature and
  source-facing tag
- Qwen2.5-0.5B and Qwen2.5-3B-Instruct briefing are reversed and
  `support-following` on both selected sides
- Qwen3-4B briefing is reversed by relation signature, with flat-side
  `conflict-following` and high-side `support-following`

## 4. Comparison Against Archive 225

| family | representative setting | high-side dominant relation signature | flat-side dominant relation signature | high-side source-facing dominant tag(s) | flat-side source-facing dominant tag(s) | clean alignment? | caveat |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `archive_v1` | strict motif `top_k=3`, `min_observer_support=3`, `min_scale_support=3`, `per_band_limit=2` | `residual_chord=3` | `residual_chord=1\|trusted_tree=2` | `conflict-following` `8/8` | `support-following` `6/8`; `non-gluing` `2/8` | yes | Archive reference from `225`; selected queue only. |
| `transcript_v1` | strict motif `top_k=3`, `min_observer_support=3`, `min_scale_support=3`, `per_band_limit=2` | `residual_chord=1\|trusted_tree=2` `6/8` | `residual_chord=3` `6/8` | `conflict-following` `6/8`; `support-following` `2/8` | `support-following` `8/8` | mixed | Source-facing tags still split high/flat in aggregate, but relation signatures mostly reverse archive; Qwen3-4B is archive-direction. |
| `briefing_v1` | strict motif `top_k=3`, `min_observer_support=3`, `min_scale_support=3`, `per_band_limit=2` | `residual_chord=1\|trusted_tree=2` `6/8` | `residual_chord=3` `6/8` | `support-following` `6/8`; `conflict-following` `2/8` | `support-following` `6/8`; `conflict-following` `2/8` | no | Source-facing tags are model-conditioned/mixed and do not reproduce archive alignment. |

The non-archive sensitivity supports a narrower family-specific reading:
archive remains the cleanest source-facing alignment surface, while transcript
and briefing are model-conditioned or mixed under the same annotation path.

The important point is not that transcript and briefing are failures. The
important point is that the exact same queue and derived annotation path does
not simply force the archive alignment onto every family.

## 5. Reading Boundary

The annotation tags describe source-facing direction only:

- `support-following`
- `conflict-following`
- `non-gluing`
- `ambiguous`

They are not answer-quality labels.
They are not semantic labels.

The derived annotation mode is an exact-anchor / phrase-rule annotation:

- exact conflict-anchor text in answer -> `conflict-following`
- exact support-anchor text in answer -> `support-following`
- no-gluing / not-warranted phrasing -> `non-gluing`
- otherwise -> `ambiguous`

It is not full human semantic annotation.

This sensitivity does not create a universal law. Archive specificity remains
bounded to the current dense-transformer artifacts and the selected
`per_band_limit = 2` source-facing queues.

No new observer mode was added. No Gate12A or Gate12B source artifact semantics
were changed.

## 6. What This Adds

This memo extends the chain:

```text
Gate12B archive flip
-> archive source queue
-> archive source-facing derived annotation
-> transcript/briefing source-facing derived annotation sensitivity
```

The added result is negative in the useful sense: transcript and briefing do
not collapse into the same clean archive source-facing alignment under the same
representative strict motif path.

That supports the bounded archive-family reading from `220` through `225` and
reduces the risk that `225` was only an annotation-rule artifact.

## 7. Short Sentence

The non-archive source-facing sensitivity supports a bounded archive-specific
read: under the same strict motif queue and exact-anchor / phrase-rule
annotation path, archive keeps a clean high-R conflict / flat-M
support-or-non-gluing alignment, while transcript and briefing mostly reverse
the relation-signature direction and remain model-conditioned or mixed.
