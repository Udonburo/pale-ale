# Gate12B Archive Candidate Source Inspection Queue

Status: source inspection queue memo draft
Role: bounded Gate12B archive candidate source-facing queue over existing Gate12A/Gate12B artifacts, not a checkpoint revision, not a release claim, not an invariant-law promotion, and not a Gate12A schema change
Date: 2026-05-06

This memo proceeds from:

- `196_GATE12_DISCRETE_CONNECTION_CONSTITUTION.md`
- `197_GATE12A_DISCRETE_CONNECTION_IMPLEMENTATION_CONTRACT.md`
- `198_GATE12A_DISCRETE_CONNECTION_AUDIT.md`
- `217_GATE12B_OBSERVER_RELATIVE_COARSE_GRAINED_CLOSURE_OPENING_MEMO.md`
- `218_GATE12B_FIRST_OBSERVER_RELATIVE_COARSE_GRAINED_CLOSURE_SMOKE_MEMO.md`
- `219_GATE12B_QWEN_2_5_0_5B_THREE_FAMILY_OBSERVER_RELATIVE_COMPARISON_MEMO.md`
- `220_GATE12B_DENSE_TRANSFORMER_FAMILY_EFFECT_EXPANSION_MEMO.md`
- `221_GATE12B_ARCHIVE_STRICT_SUPPORT_SENSITIVITY_MEMO.md`
- `222_GATE12B_ARCHIVE_OBSERVER_SCOPE_EXPANSION_SENSITIVITY_MEMO.md`
- `223_GATE12B_MOTIF_OBSERVER_SPECIFICITY_CHECK.md`

## 0. Scope

This memo opens the source-level inspection phase for the archive-family
Gate12B signal.

The question is narrow:

```text
Do the archive flat-side M candidates and high-side R candidates look different
when returned to source text, relation path, and answer-support geometry?
```

This pass uses the representative strict motif setting:

- `family = archive_v1`
- `observer_mode_set = cycle_motif_expansion_v1`
- `top_k = 3`
- `min_observer_support = 3`
- `min_scale_support = 3`

It checks the current dense-transformer archive set:

- `Qwen/Qwen2.5-0.5B`
- `Qwen/Qwen2.5-3B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`
- `Qwen/Qwen3-4B`

It does not:

- run model inference
- change Gate12A or Gate12B source artifacts
- add a new observer mode
- treat any candidate row as an answer-quality label
- turn exact text matching into a semantic evaluator

## 1. Added Queue Builder

This pass adds:

- `tools/build_gate12b_source_inspection_queue.py`

The tool reads:

- Gate12B `invariant_signature_candidates.jsonl`
- Gate12B `manifest.json`
- matching Gate12A text-surface `triangle_text_surface_joined.jsonl`
- matching Gate12A text-surface `manifest.json`

and emits:

- `gate12b_source_inspection_queue.csv`
- `gate12b_source_inspection_queue.jsonl`
- `gate12b_source_inspection_queue.md`
- `gate12b_source_inspection_queue_status.json`
- `manifest.json`
- `checksums.json`

The output is source-facing. It carries prompt, answer, support anchor, conflict
anchor, relation path, anchor-qualified path, observer support, and scale
support into one review queue.

The exact anchor text indicators are only inspection helpers:

- `answer_contains_support_anchor_text`
- `answer_contains_conflict_anchor_text`

They are exact normalized substring checks. They are not semantic judgments.

## 2. Run

The queue artifact is:

- `runs/gate12b_archive_candidate_source_inspection_queue_motif_topk3/`

The run used:

- `per_band_limit = 2`
- `4` archive cases
- `2` flat rows per case
- `2` high-tension rows per case

Status:

| field | value |
| --- | ---: |
| cases | 4 |
| queue rows | 16 |
| flat rows | 8 |
| high-tension rows | 8 |
| answer contains support anchor text | 6 |
| answer contains conflict anchor text | 8 |
| checksum mismatches | 0 |
| builder script sha256 | `8b0ac133af46e08c60715e44556200cdecdf3d9d74ba09e95918f312e0f5bb5e` |

## 3. Selected Rows

| case | side | cycles | sample | relation signature | answer anchor text read |
| --- | --- | --- | --- | --- | --- |
| `Qwen/Qwen2.5-0.5B` | flat | `triangle:000295`, `triangle:000294` | `sample_000116` | `residual_chord=1\|trusted_tree=2` | support text present, conflict text absent |
| `Qwen/Qwen2.5-0.5B` | high | `triangle:000124`, `triangle:000126` | `sample_000048` | `residual_chord=3` | conflict text present, support text absent |
| `Qwen/Qwen2.5-3B-Instruct` | flat | `triangle:000224`, `triangle:000225` | `sample_000081` | `residual_chord=1\|trusted_tree=2` | support text absent, conflict text absent |
| `Qwen/Qwen2.5-3B-Instruct` | high | `triangle:000142`, `triangle:000140` | `sample_000052` | `residual_chord=3` | conflict text present, support text absent |
| `meta-llama/Llama-3.2-3B-Instruct` | flat | `triangle:000018`, `triangle:000019` | `sample_000010` | `residual_chord=1\|trusted_tree=2` | support text present, conflict text absent |
| `meta-llama/Llama-3.2-3B-Instruct` | high | `triangle:000070`, `triangle:000068` | `sample_000034` | `residual_chord=3` | conflict text present, support text absent |
| `Qwen/Qwen3-4B` | flat | `triangle:000019`, `triangle:000018` | `sample_000010` | `residual_chord=1\|trusted_tree=2` | support text present, conflict text absent |
| `Qwen/Qwen3-4B` | high | `triangle:000092`, `triangle:000094` | `sample_000040` | `residual_chord=3` | conflict text present, support text absent |

## 4. First Source-Level Read

The first queue read is directionally consistent with the archive flip:

- high-side `residual_chord=3` rows are conflict-facing in all selected cases
- flat-side `residual_chord=1|trusted_tree=2` rows are not conflict-facing in
  any selected case
- flat-side rows are usually support-facing by exact anchor text
- the Qwen2.5-3B flat exception is an abstention/no-gluing surface, not a
  conflict-following surface

The Qwen2.5-3B flat rows say:

```text
Given the ledger split, no direct path conclusion from Thorn to Grove is
warranted across separate ledgers. The notes cannot be glued into one
transitive conclusion.
```

The support anchor for those rows is:

```text
a directed path exists from Thorn to Grove
```

So this row does not match the support anchor text exactly, but it also does not
follow a conflict anchor.

## 5. Reading Boundary

This memo earns only a queue-level sentence:

- in the selected archive source-inspection queue, high-side R candidates are
  consistently conflict-facing, while flat-side M candidates are support-facing
  or non-gluing and never conflict-facing by the exact anchor-text helper

It does not earn:

- a full semantic phenotype claim
- a model-quality claim
- a universal archive law
- that exact substring matching is enough for final source inspection
- that source-level inspection is complete

## 6. Next Inspection Step

The next useful step is manual or semi-structured annotation of the 16 queued
rows with a small source-facing tag set:

- support-following
- conflict-following
- non-gluing / abstention
- ambiguous

That should be done without adding new observer modes.

## 7. Short Sentence

The first archive source-inspection queue supports the Gate12B archive flip at
the source surface: selected high-side `residual_chord=3` candidates are
conflict-facing in `8/8` rows, while selected flat-side
`residual_chord=1|trusted_tree=2` candidates are support-facing in `6/8` rows
and non-gluing in the remaining `2/8`.
