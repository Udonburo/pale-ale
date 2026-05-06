# Gate12B Observer-Relative Closure Closeout Memo

Status: Gate12B closeout memo draft
Role: bounded closeout for Gate12B observer-relative coarse-grained closure, source-facing queue, and source-facing annotation; not a checkpoint revision, not a release claim, not an invariant-law promotion, not a model-quality benchmark, and not a Gate12A/Gate12B schema change
Date: 2026-05-06

This memo closes:

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

## 0. Scope

This memo closes Gate12B as a bounded artifact-study layer over existing
Gate12A outputs.

The closeout question is:

```text
What does Gate12B earn, and what must remain outside the claim boundary?
```

The answer is deliberately limited. Gate12B earns a reproducible
archive-family closure-signature result over the current dense-transformer
artifact set. It does not earn a universal reasoning law, model-quality score,
or checkpoint claim.

## 1. What Gate12B Added

Gate12B added a read-only secondary audit over existing Gate12A artifacts.

The primitive was:

```text
observer x scale x admissible_gauge_transform
```

The implemented vocabulary kept the local reparameterization boundary narrow:

- basis-preserving local reparameterization
- projector-level invariant
- transport-level compatibility invariant
- gauge-stable closure signature

Gate12B did not:

- change Gate12A semantics
- change Gate12A or Gate12B thresholds
- change Gate12A or Gate12B classifications
- change Gate12A or Gate12B schemas
- require new model inference
- require GPU work

The Gate12B line was CPU-only artifact reading, summarization, queue building,
and source-facing derived annotation.

## 2. Evidence Chain

The chain is:

- `217`: opened Gate12B as observer-relative coarse-grained closure over
  existing Gate12A artifacts
- `218`: ran the first smoke and established the runner artifact shape
- `219`: compared Qwen2.5-0.5B across transcript, briefing, and archive
- `220`: expanded to the dense-transformer set and identified archive as the
  bounded family-conditioned surface
- `221`: checked archive strict scale-support sensitivity
- `222`: checked archive observer-scope expansion sensitivity
- `223`: introduced `cycle_motif_expansion_v1` as an explicit sensitivity and
  showed that it restores strict archive support without forcing all families
  into the archive direction
- `224`: returned selected archive candidates to source-facing queue rows
- `225`: derived source-facing annotations for the archive queue
- `226`: ran the same source-facing queue and annotation path over transcript
  and briefing as a non-archive sensitivity

## 3. Earned Bounded Claims

Gate12B earns this bounded claim:

```text
Gate12B earns a bounded archive-family closure-signature result: across the
current dense-transformer archive surfaces, high-side `residual_chord=3` and
flat-side `residual_chord=1|trusted_tree=2` form a repeated
relation-signature flip that survives stricter scale support, motif
observer-scope expansion, and the current basis-preserving local
reparameterization check; selected source-facing rows align high-side R with
conflict-following and flat-side M with support-following or non-gluing.
```

The non-archive source-facing sensitivity does not collapse transcript and
briefing into the same clean archive source-facing alignment, supporting a
bounded family-specific reading rather than an annotation-rule-only artifact.

## 4. Not-Earned Claims

Gate12B does not earn:

- a universal law of LLM reasoning
- a model-quality benchmark
- a correctness classifier
- a claim that high residual always means bad answer
- a claim that flat always means correct answer
- a physical gauge invariant
- Yang-Mills / Chern / principal-bundle language
- a claim about Transformer weights in general
- a checkpoint or release claim
- a replacement for Gate12A

It also does not open:

- spin network language
- TQFT language
- action or Lagrangian language
- connection Laplacian terminology

Those remain outside this closeout.

## 5. Main Result

The main Gate12B result is the archive-family relation-signature flip:

```text
archive_v1:
  high = residual_chord=3
  flat = residual_chord=1|trusted_tree=2
```

This repeated across:

- `Qwen/Qwen2.5-0.5B`
- `Qwen/Qwen2.5-3B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`
- `Qwen/Qwen3-4B`

The transcript and briefing families did not become archive-like under the
same motif observer sensitivity. In `223`, transcript and briefing mostly
showed the reverse relation-signature direction and remained
model-conditioned. In `226`, their source-facing derived annotations also
remained mixed or model-conditioned rather than reproducing the clean archive
alignment.

## 6. Sensitivity Results

Archive strict scale support preserved the relation-signature flip at:

- `top_k = 1`
- `top_k = 3`
- `top_k = 5`

The default `core_v1` observer set with strict observer support caused archive
candidates to vanish at the stricter setting, which was a useful boundary
rather than a failure.

`cycle_motif_expansion_v1` restored strict observer plus scale support for the
archive line. It did not force all families into the archive direction:

- archive kept high=`residual_chord=3` and
  flat=`residual_chord=1|trusted_tree=2`
- transcript and briefing mostly showed the reverse direction under the same
  `cycle_motif_expansion_v1` sensitivity

The reparameterization checks were mostly quiet.

The known caveat remains the Qwen3-4B briefing threshold-boundary case:

- `triangle:000358`
- band transition: `tense -> flat`
- max residual delta: `1.6653345369377348e-16`
- not an invariant candidate
- not a gauge-stable candidate

This is recorded as a threshold-boundary caveat, not candidate-level
instability.

## 7. Source-Facing Results

`224` built the selected archive source inspection queue:

- `16` rows
- `8` flat rows
- `8` high-tension rows
- `per_band_limit = 2`

`225` derived source-facing annotations for that archive queue:

- high-side `residual_chord=3` -> `conflict-following` in `8/8`
- flat-side `residual_chord=1|trusted_tree=2` -> `support-following` in `6/8`
- flat-side `residual_chord=1|trusted_tree=2` -> `non-gluing` in `2/8`

`226` ran the same queue and annotation path on transcript and briefing:

| family | rows | support-following | conflict-following | non-gluing | ambiguous | reading |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `archive_v1` | 16 | 6 | 8 | 2 | 0 | clean archive high-R conflict / flat-M support-or-non-gluing |
| `transcript_v1` | 16 | 10 | 6 | 0 | 0 | relation signatures mostly reverse archive; source tags split high/flat in aggregate |
| `briefing_v1` | 16 | 12 | 4 | 0 | 0 | relation signatures mostly reverse archive; source tags are model-conditioned/mixed |

This supports the bounded reading:

- archive is the cleanest source-facing alignment surface in the current
  selected queue set
- transcript and briefing do not reproduce the full archive relation-plus-source
  alignment under the same annotation path

The source-facing tags are not answer-quality labels.

## 8. Gauge / Reparameterization Boundary

Gate12B uses bounded gauge language only in the Gate12 sense:

- basis-preserving local reparameterization
- projector-level invariant
- transport-level compatibility invariant
- gauge-stable closure signature

This is not a claim of physical gauge theory.

Gate12B does not introduce:

- Yang-Mills
- principal bundle
- Chern class
- SU(2)
- U(1)
- physical gauge field
- full gauge symmetry

The local reparameterization check is an artifact-level stability check, not a
physics claim.

## 9. Paper-Readiness

Gate12B is paper-ready only as a bounded empirical artifact report after this
closeout:

- `226` is recorded
- this closeout is recorded
- generated run artifacts remain local unless packaged intentionally later
- paper claims remain bounded

Suggested framing:

```text
This is not a universal interpretability law. It is a reproducible, bounded
artifact study reporting an observer-relative, coarse-grained
closure-signature phenomenon in existing LLM audit artifacts.
```

## 10. Recommended Paper Claim

Recommended claim:

```text
We report a bounded repeated archive-family closure signature in discrete
inference transport artifacts. Across four dense-transformer model lines,
archive-family Gate12B surfaces repeatedly place high-tension candidates on
`residual_chord=3` and flat candidates on
`residual_chord=1|trusted_tree=2`. The signal survives stricter scale support,
motif observer-scope expansion, and the current basis-preserving local
reparameterization check. A selected source-facing queue aligns high-side R
candidates with conflict-following rows and flat-side M candidates with
support-following or non-gluing rows. Non-archive source-facing sensitivity
does not reproduce the same clean archive alignment under the same queue and
annotation path.
```

## 11. Next Work

Recommended next steps:

- write the paper outline
- optionally package selected run artifacts intentionally
- optionally expand source-facing annotation with a larger `per_band_limit`
- optionally run a supplied/manual annotation pass
- do not add new observers before writing the paper outline
- do not open connection Laplacian / action / Yang-Mills language in this
  closeout line

## 12. Short Sentence

Gate12B closes with a bounded archive-family result: the current
dense-transformer archive surfaces show a repeated relation-signature flip
that survives observer/scale/reparameterization sensitivity and aligns with
selected source-facing annotations, while remaining explicitly non-universal,
non-quality-scoring, and artifact-bounded.
