# Gate12B Paper Outline and Claim Boundary Memo

Status: Gate12B paper outline and claim-boundary memo draft
Role: bounded paper-readiness outline for Gate12B observer-relative closure, not a paper draft, not a checkpoint revision, not a release claim, not an invariant-law promotion, not a model-quality benchmark, and not a Gate12A/Gate12B schema change
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
- `226_GATE12B_NONARCHIVE_SOURCE_FACING_ANNOTATION_SENSITIVITY.md`
- `227_GATE12B_OBSERVER_RELATIVE_CLOSURE_CLOSEOUT_MEMO.md`

## 0. Scope

This memo turns the closed Gate12B evidence line into a paper outline and a
claim boundary.

It does not add experiments, observers, metrics, or theory vocabulary. It fixes
what the paper can say, what evidence supports that statement, and what the
paper must not imply.

The intended next sequence is:

```text
228 paper outline / claim boundary
229 evidence package manifest
230 paper draft v0
```

The core posture remains:

```text
bounded artifact study, not universal interpretability law
```

## 1. One-Sentence Claim

Recommended one-sentence claim:

```text
We report a bounded archive-family closure-signature phenomenon in existing
LLM audit artifacts: across four dense-transformer model lines, archive-family
Gate12B surfaces repeatedly place high-tension candidates on residual_chord=3
and flat candidates on residual_chord=1|trusted_tree=2; the signal survives
observer/scale/reparameterization sensitivity and aligns with selected
source-facing annotations, while transcript and briefing sensitivities do not
reproduce the same clean archive alignment.
```

This sentence is the paper's ceiling. The draft can explain and qualify it, but
should not quietly expand beyond it.

## 2. Claim

The paper claim has four parts.

First, Gate12B reports an archive-family closure signature:

```text
archive_v1:
  high-tension side = residual_chord=3
  flat side = residual_chord=1|trusted_tree=2
```

Second, the signature repeats across the current four dense-transformer model
lines:

- `Qwen/Qwen2.5-0.5B`
- `Qwen/Qwen2.5-3B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`
- `Qwen/Qwen3-4B`

Third, the signal survives the current Gate12B sensitivity checks:

- stricter scale support
- motif observer-scope expansion
- current basis-preserving local reparameterization check

Fourth, the selected source-facing annotation layer supports the structural
read:

```text
archive high-side residual_chord=3 -> conflict-following 8/8
archive flat-side residual_chord=1|trusted_tree=2 -> support-following 6/8
archive flat-side residual_chord=1|trusted_tree=2 -> non-gluing 2/8
```

The non-archive sensitivity is part of the defense:

```text
transcript_v1:
  high-side dominant relation signature = residual_chord=1|trusted_tree=2
  flat-side dominant relation signature = residual_chord=3

briefing_v1:
  high-side dominant relation signature = residual_chord=1|trusted_tree=2
  flat-side dominant relation signature = residual_chord=3
```

This means the exact same queue and derived source-facing annotation path does
not simply make transcript and briefing look archive-like.

## 3. Method

The method section should be framed as a read-only audit pipeline.

Inputs:

- Gate12A `manifest.json`
- Gate12A `explicit_triangle_cycle_registry.jsonl`
- Gate12A `triangle_holonomy_registry.jsonl`
- Gate12A `transport_relation_registry.jsonl`
- Gate12A triangle text-surface audit outputs when source-facing rows are used

Gate12B audit primitive:

```text
observer x scale x bounded local reparameterization
```

Operational layers:

- observer-relative closure views
- coarse-grained scale views
- basis-preserving local reparameterization checks
- invariant signature candidate extraction
- motif observer sensitivity with `cycle_motif_expansion_v1`
- source inspection queue
- exact-anchor / phrase-rule source-facing derived annotation

Representative strict motif setting:

```text
observer_mode_set = cycle_motif_expansion_v1
top_k = 3
min_observer_support = 3
min_scale_support = 3
per_band_limit = 2
```

The paper must keep this as artifact analysis. It should not describe the
source-facing annotation as human semantic annotation unless a supplied manual
annotation file is introduced in later work.

## 4. Evidence Chain

| evidence step | role in paper |
| --- | --- |
| `217` | Opens Gate12B as observer-relative coarse-grained closure over existing Gate12A artifacts. |
| `218` | Establishes the first runner smoke and artifact shape. |
| `219` | Shows the three-family comparison on Qwen2.5-0.5B. |
| `220` | Expands to dense-transformer family effect and identifies archive as the stable family-conditioned surface. |
| `221` | Shows archive strict scale-support sensitivity. |
| `222` | Shows archive observer-scope expansion sensitivity. |
| `223` | Shows motif observer expansion restores strict support for archive without forcing transcript/briefing into archive direction. |
| `224` | Returns selected archive candidates to source-facing inspection queue rows. |
| `225` | Adds archive source-facing derived annotation: high R conflict, flat M support-or-non-gluing. |
| `226` | Adds transcript/briefing source-facing sensitivity and shows non-archive does not reproduce the clean archive alignment. |
| `227` | Closes Gate12B with bounded claims and non-claims. |

This chain should become the paper's Results spine.

## 5. Paper Outline

Suggested title:

```text
Observer-Relative Closure Signatures in Existing LLM Audit Artifacts
```

Suggested section plan:

1. Abstract
2. Introduction
3. Artifact Setting
4. Gate12B Method
5. Archive-Family Closure Signature
6. Observer, Scale, and Reparameterization Sensitivity
7. Source-Facing Queue and Derived Annotation
8. Non-Archive Sensitivity
9. Threats and Limits
10. Reproducibility Map
11. Conclusion

Abstract should state:

- the study is bounded and artifact-based
- the main result is archive-family only
- the non-archive sensitivity does not reproduce the clean archive alignment
- the result is not a model-quality benchmark

Introduction should avoid broad claims about LLM reasoning. It should motivate
why closure structure in replay artifacts can be studied without converting it
into a scalar score.

Method should emphasize:

- read-only secondary audit
- no new model inference
- no Gate12A/Gate12B schema change
- explicit sensitivity axes

Results should lead with the archive signature, then show why the signature is
not merely produced by the motif observer or annotation rule.

Conclusion should return to the one-sentence claim and stop there.

## 6. Figures / Tables

Recommended paper figures and tables:

| item | content | source |
| --- | --- | --- |
| Figure 1 | Gate12A -> Gate12B -> source queue -> source-facing annotation pipeline | `217`, `224`, `225` |
| Table 1 | Dense-transformer family comparison: archive vs transcript vs briefing relation signatures | `220`, `223` |
| Table 2 | Archive strict support sensitivity across `top_k=1/3/5` | `221`, `222`, `223` |
| Table 3 | Source-facing annotation summary: archive, transcript, briefing | `225`, `226` |
| Table 4 | Non-claims and boundary conditions | `227`, this memo |
| Table 5 | Caveats: Qwen3-4B briefing threshold-boundary case and annotation limits | `223`, `226`, `227` |

The most important table is the family comparison:

| family | high-side dominant relation signature | flat-side dominant relation signature | source-facing reading |
| --- | --- | --- | --- |
| `archive_v1` | `residual_chord=3` | `residual_chord=1\|trusted_tree=2` | clean high conflict / flat support-or-non-gluing |
| `transcript_v1` | `residual_chord=1\|trusted_tree=2` | `residual_chord=3` | mixed; not the clean archive alignment |
| `briefing_v1` | `residual_chord=1\|trusted_tree=2` | `residual_chord=3` | mixed/model-conditioned; not the clean archive alignment |

## 7. Non-Claims

The paper must explicitly say that the result is not:

- a universal law of LLM reasoning
- a model-quality benchmark
- a correctness classifier
- a claim that high residual always means bad answer
- a claim that flat always means correct answer
- a physical gauge invariant
- a claim about model weights in general
- a checkpoint or release claim
- a replacement for Gate12A

The paper must not introduce:

- Yang-Mills
- principal bundle
- Chern class
- SU(2)
- U(1)
- physical gauge field
- full gauge symmetry
- spin network
- TQFT
- action or Lagrangian language
- connection Laplacian terminology

If these terms appear at all, they should appear only in boundary lists like
this one.

## 8. Threats / Limits

Key limits:

- The study uses existing local artifacts; a future paper package must make
  the evidence package explicit.
- The current source-facing queue uses `per_band_limit = 2`.
- The derived annotation is exact-anchor / phrase-rule only.
- Source-facing tags are not semantic labels.
- Source-facing tags are not answer-quality labels.
- The current dense-transformer set has four model lines.
- The archive result is family-bounded and artifact-bounded.
- Transcript and briefing are sensitivity surfaces, not failed runs.
- The current reparameterization check is bounded to the evaluated
  basis-preserving local transform.
- The Qwen3-4B briefing `triangle:000358` caveat is a threshold-boundary
  caveat, not candidate-level instability.

The annotation-rule threat is specifically addressed by `226`: transcript and
briefing do not reproduce the same clean archive alignment under the same
queue and derived annotation path.

## 9. Reproducibility Map

| paper claim fragment | evidence memo | helper / artifact class |
| --- | --- | --- |
| Gate12B is a read-only secondary audit | `217`, `218`, `227` | `tools/run_gate12b_observer_relative_coarse_grained_closure.py` |
| Archive high=`residual_chord=3`, flat=`residual_chord=1\|trusted_tree=2` repeats across four model lines | `220`, `221`, `222`, `223` | Gate12B run directories and `tools/summarize_gate12b_runs.py` |
| Motif observer expansion does not force all families into archive direction | `223` | `cycle_motif_expansion_v1` Gate12B runs |
| Archive source queue returns selected candidates to source surfaces | `224` | `tools/build_gate12b_source_inspection_queue.py` |
| Archive source-facing annotation aligns high R with conflict and flat M with support-or-non-gluing | `225` | `tools/annotate_gate12b_source_inspection_queue.py` |
| Non-archive source-facing sensitivity does not reproduce clean archive alignment | `226` | transcript/briefing queue and annotation artifacts |
| Claim and non-claim boundary | `227`, `228` | workstream memos |

The next memo, `229`, should not dump all `runs/` artifacts into git. It should
define a paper evidence package manifest: which local artifacts are needed,
which checksums identify them, and how they map to the tables above.

## 10. Recommended Abstract Skeleton

Possible abstract shape:

```text
We study closure signatures in existing LLM audit artifacts using Gate12B, a
read-only secondary audit over Gate12A discrete inference transport outputs.
Across four dense-transformer model lines, archive-family surfaces repeatedly
place high-tension candidates on residual_chord=3 and flat candidates on
residual_chord=1|trusted_tree=2. The signal survives stricter scale support,
motif observer-scope expansion, and the current basis-preserving local
reparameterization check. A selected source-facing queue aligns high-side R
candidates with conflict-following rows and flat-side M candidates with
support-following or non-gluing rows. Transcript and briefing sensitivities do
not reproduce the same clean archive alignment. We present this as a bounded
artifact study, not a universal interpretability law or model-quality
benchmark.
```

## 11. 229 Handoff

`229` should be:

```text
229_GATE12B_PAPER_EVIDENCE_PACKAGE_MANIFEST.md
```

It should list:

- required Gate12B run directories
- required source inspection queue artifacts
- required source-facing annotation artifacts
- summary artifacts
- builder script hashes
- checksum expectations
- which paper table each artifact supports

It should keep generated `runs/` artifacts uncommitted unless a deliberate
package format is chosen.

## 12. Short Sentence

Gate12B paper readiness should start from one bounded sentence: the current
dense-transformer archive artifacts show a repeated high-R / flat-M closure
signature that survives the recorded sensitivities and aligns with selected
source-facing rows, while non-archive sensitivities do not reproduce the same
clean archive alignment.
