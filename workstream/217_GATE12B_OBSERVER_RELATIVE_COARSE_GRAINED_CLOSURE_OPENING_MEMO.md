# Gate12B Observer-Relative Coarse-Grained Closure Opening Memo

Status: opening memo / implementation-direction memo
Role: Gate12B read-only secondary audit direction over existing Gate12A artifacts, not a Gate12A schema change, not graph-wide smoothing, not a scalar score doctrine, and not external physics terminology in code
Date: 2026-05-05

This memo proceeds from:

- `195_GATE11_ADMISSIBILITY_MEMBRANE_FREEZE.md`
- `196_GATE12_DISCRETE_CONNECTION_CONSTITUTION.md`
- `197_GATE12A_DISCRETE_CONNECTION_IMPLEMENTATION_CONTRACT.md`
- `198_GATE12A_DISCRETE_CONNECTION_AUDIT.md`
- `199_GATE12A_ANCHOR_RICH_CLOSURE_TENSION_EMPIRICAL_MEMO.md`
- `200_GATE12A_TRANSCRIPT_V1_ANCHOR_RICH_CLOSURE_TENSION_REPLICATION_MEMO.md`
- `201_GATE12A_BRIEFING_V1_ANCHOR_RICH_CLOSURE_TENSION_EMPIRICAL_MEMO.md`
- `202_GATE12A_TRANSCRIPT_V1_VS_BRIEFING_V1_ANCHOR_RICH_CLOSURE_TENSION_COMPARISON_MEMO.md`

## 0. Scope

Gate12B opens as a read-only secondary audit over the existing Gate12A artifact
surface.

Gate12B does:

- add observer-relative views over existing Gate12A triangle closure rows
- add coarse-grained scale views over the same closure rows
- add a bounded admissible gauge-stability check as an invariant test
- preserve Gate12A semantics, thresholds, classifications, and artifact schemas
- emit new Gate12B artifacts without rewriting the Gate12A source artifacts

Gate12B does not:

- change Gate12A transport, holonomy, or residual definitions
- reopen Gate12A classification semantics
- collapse closure behavior into one scalar score
- use graph-wide smoothing as the public read
- promote external physics terminology into code

## 1. Public Question

The Gate12B question is:

- `if observer, scale, and admissible local representation are varied, does any closure-defect signature remain stable?`

The short working form is:

- `observer changes may move the defect distribution; coarse-graining may reduce local variance; admissible local representation changes should not change projector-level closure signatures.`

## 2. Three-Axis Primitive

The Gate12B primitive is:

- `observer x scale x admissible_gauge_transform`

The axes are intentionally narrow.

### 2.1 Observer

Initial observer views are:

- `all_edges`
- `anchor_qualified`
- `residual_chord_heavy`
- `relation_kind_conditioned`

These are views over explicit Gate12A rows.
They are not new hidden graph relations.

### 2.2 Scale

Initial scale views are:

- `triangle`
- `relation_kind_band`
- `anchor_policy_band`
- `residual_quantile_band`

These are coarse-grained summaries of existing triangle closure rows.
They are not a new graph-wide operator.

### 2.3 Admissible Gauge Transform

The admissible gauge transform is limited to:

- basis-preserving local reparameterization
- projector-level invariant checks
- transport-level compatibility invariant checks

The safe Gate12B definition is:

- `admissible local gauge change = a local basis / representation change that preserves projector/span identity and should not alter projector-level closure signatures.`

## 3. Input Artifacts

The first Gate12B runner reads exactly one existing Gate12A run directory.

Required inputs:

- `manifest.json`
- `explicit_triangle_cycle_registry.jsonl`
- `triangle_holonomy_registry.jsonl`
- `transport_relation_registry.jsonl`

Optional input:

- `transport_operator_arrays.npz`

If transport arrays are present, the first gauge-stability check may apply a
deterministic local coordinate reversal to the active transport blocks and
compare pre/post closure residual bands.

If transport arrays are absent, the audit remains registry-only and records that
the nontrivial array-level reparameterization was not evaluated.

## 4. Output Artifacts

The first Gate12B runner emits:

- `manifest.json`
- `observer_scale_closure_matrix.csv`
- `observer_scale_closure_matrix.json`
- `invariant_signature_candidates.jsonl`
- `gauge_stability_matrix.csv`
- `gauge_stability_summary.json`
- `gauge_variant_signature_candidates.jsonl`
- `gate12b_observer_relative_coarse_grained_closure.md`
- `checksums.json`

These are Gate12B artifacts only.
They do not mutate or replace the Gate12A artifact family.
The output directory must be separate from the input Gate12A artifact directory,
and the runner must reject an aliased `out_dir` before writing any files.

`invariant_signature_candidates.jsonl` must not be a triangle-only top-k list.
Candidate rows require:

- support across independent observer scopes, where observer views with identical
  cycle membership count as one observer scope
- support across multiple scale modes
- at least one non-triangle coarse scale support

`gauge_variant_signature_candidates.jsonl` may be emitted only when a nontrivial
array-level admissible reparameterization was evaluated.
If `transport_operator_arrays.npz` is absent, the registry-only path may record
that the gauge check was skipped, but it must not promote gauge-stable
candidates.

## 5. Hypotheses

H1:

- `Closure-defect magnitudes are observer-relative, but a subset of defect signatures remains stable under coarse-graining.`

H1-A:

- `Changing the observer changes the defect distribution.`

H1-B:

- `Coarse-graining reduces local variance but preserves conflict-aligned ordering.`

H1-C:

- `Invariant signature candidates survive across multiple observer x scale views.`

H1-D:

- `Projector-level closure signatures remain stable under admissible basis-preserving local reparameterizations.`

## 6. Non-Claims

Gate12B does not claim:

- high residual means bad answer
- low residual means good answer
- one closure scalar is sufficient
- observer-relative movement is itself a failure
- all future families share one invariant phenotype law
- a separate physics formalism has been implemented

## 7. Done Conditions

The first Gate12B implementation is done when:

- it reads the existing Gate12A artifact family without modifying it
- it emits an observer x scale matrix
- it applies at least one admissible gauge transform when transport arrays are present
- it compares pre/post band or status stability
- it emits invariant signature candidates as JSONL
- it preserves existing Gate12A semantics and classifications
- its focused Python tests pass
- local-only notes remain outside tracked git status

## 8. Short Sentence

Gate12B asks whether observer-relative closure defects contain signatures that
survive coarse-graining and admissible basis-preserving local reparameterization,
while leaving the Gate12A source artifacts unchanged.
