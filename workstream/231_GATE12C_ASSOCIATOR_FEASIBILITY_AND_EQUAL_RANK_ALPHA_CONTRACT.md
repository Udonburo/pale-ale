# Gate12C Associator Feasibility and Equal-Rank Alpha Contract

Status: spec-only implementation handoff
Role: additive read-only Gate12A artifact audit contract for compressed-overlap parenthesization sensitivity, not a Gate12A or Gate12B schema revision, not a general replay-graph triangle audit, not a rectangular-rank associator contract, and not a physical nonassociativity claim
Date: 2026-06-24

This memo proceeds from:

- `197_GATE12A_DISCRETE_CONNECTION_IMPLEMENTATION_CONTRACT.md`
- `198_GATE12A_DISCRETE_CONNECTION_AUDIT.md`
- `217_GATE12B_OBSERVER_RELATIVE_COARSE_GRAINED_CLOSURE_OPENING_MEMO.md`
- `228_GATE12B_PAPER_OUTLINE_AND_CLAIM_BOUNDARY.md`
- `tools/run_gate12a_discrete_connection_audit.py`
- `tools/run_gate12b_observer_relative_coarse_grained_closure.py`

## 0. Executive Decision

Gate12C is split into two ordered phases.

1. `Gate12C-0` is a feasibility preflight over existing Gate12A artifacts.
2. `Gate12C-1` is an equal-rank alpha audit only if the preflight exposes a usable empirical surface.

The governing sentence is:

> Gate12C-0 first verifies whether Gate12A-defined residual-bearing explicit triangles contain a declared minimum number of equal-rank, common-rank `r >= 2` cycles with stable nontrivial SVD cuts. Gate12C-1 alpha then measures parenthesization sensitivity on that equal-rank surface only. Rectangular rank-mismatch associators are deferred.

The core observable is not ordinary matrix nonassociativity. Ordinary matrix multiplication remains associative and is a required null control.

The alpha question is narrower:

> Does a frozen low-rank recomposition law applied between binary overlap compositions produce reproducible parenthesization sensitivity on Gate12A-defined residual-bearing explicit triangles?

## 1. Exact Scope Boundary

### 1.1 Gate12A cycle surface

Gate12C alpha does not operate on all replay-graph triangles.

Its input cycle surface is exactly the Gate12A v1 surface:

- explicit directed cycles of length `3`
- already materialized in `explicit_triangle_cycle_registry.jsonl`
- all three transport-bearing legs explicitly present
- at least one leg with `relation_kind = residual_chord`
- no generic cycle mining
- no invented closure edge

The required public wording is:

> Gate12A-defined residual-bearing explicit triangles

The following broader phrases are forbidden in the alpha claim surface:

- all replay graph triangles
- all artifact graph triangles
- generic replay-graph cycles
- arbitrary triple overlaps

### 1.2 Equal-rank-only alpha

`Gate12C-1` alpha is restricted to cycles satisfying all of the following:

- `holonomy_status = defined`
- every leg has `transport_case = equal_rank_orthogonal`
- all three nodes share one common positive rank `r`
- `r >= 2`

For this memo, `cycle_rank` means only:

> the common node rank of a Gate12A-defined equal-rank cycle

It must not be emitted for a rectangular or rank-mismatch cycle.

Rectangular overlaps and partial-isometry chains are mathematically composable in some cases, but their admissible compression ranks are root- and inner-product-dependent. They are explicitly deferred from alpha.

### 1.3 Read-only boundary

Gate12C must:

- read existing Gate12A artifacts
- write to a separate output directory
- leave the source Gate12A directory unchanged
- perform no model inference
- alter no Gate12A threshold, row, array, manifest, or classification
- alter no Gate12B threshold, row, array, manifest, candidate, or classification

Gate12C is a sibling read-only audit over Gate12A outputs. It is not a Gate12B subroutine and must not consume Gate12B candidate rows in its core computation.

## 2. Operational Objects

### 2.1 Node-local basis

For each defined Gate12A node `v`, read the active basis factor

```text
U_v in R^(d x r)
```

from:

```text
node_local_object_registry.jsonl
node_local_object_arrays.npz
```

The public projector remains implicit:

```text
P_v = U_v U_v^T
```

Dense projectors are not required as stored artifacts.

### 2.2 Raw overlap

For a directed edge `source -> target`, reconstruct the raw overlap:

```text
C_(target<-source) = U_target^T U_source
```

Gate12C uses this raw overlap as its primary composition input.

### 2.3 Polar transport remains separate

Gate12A stores the polar or partial-isometry transport surface in:

```text
transport_operator_arrays.npz::transport_matrix_local
```

Gate12A/Gate12B holonomy is computed from those transport blocks.

Gate12C must not rename a raw-overlap quantity as Gate12A holonomy. The two operator surfaces remain distinct:

```text
C : raw overlap operator
R : Gate12A polar transport operator
```

### 2.4 Traversal order

`edge_id_path` is a canonicalized identifier set and must not be treated as traversal order.

The ordered edge path must be reconstructed from consecutive pairs in `node_id_path` and the source/target ids in `transport_relation_registry.jsonl`, following the existing Gate12B discipline.

## 3. Required Input Artifact Family

Gate12C-0 requires:

```text
manifest.json
node_local_object_registry.jsonl
node_local_object_arrays.npz
transport_relation_registry.jsonl
transport_operator_arrays.npz
explicit_triangle_cycle_registry.jsonl
triangle_holonomy_registry.jsonl
```

Required arrays are:

```text
node_local_object_arrays.npz
  basis_factor
  rank_active

transport_operator_arrays.npz
  transport_matrix_local
  overlap_singular_values
  active_rank
```

The implementation is contract-feasible because the Gate12A artifact contract carries enough information to reconstruct `C`.

Empirical feasibility is unknown until Gate12C-0 runs on real artifact directories. This memo makes no claim that rank-2 or rank-3 cycles are abundant.

## 4. Gate12C-0 Feasibility Preflight

### 4.1 Purpose

Gate12C-0 answers whether the existing artifact surface can support a nontrivial equal-rank alpha audit.

It must not compute or promote Gate12B high/flat overlays, source-facing interpretations, or physical language.

### 4.2 Declared minimum

The preflight must accept an explicit caller-supplied minimum:

```text
--min-eligible-cycles N
```

The runner must not silently invent a scientific sufficiency threshold.

The manifest and status output must record the exact `N` used.

At minimum, two separate statuses must be emitted:

```text
contract_feasibility_status
empirical_surface_status
```

Suggested values:

```text
contract_feasibility_status:
  pass
  fail_missing_artifact
  fail_reconstruction_mismatch
  fail_ordinary_associativity_null

empirical_surface_status:
  pass_declared_minimum
  fail_below_declared_minimum
  fail_no_nontrivial_equal_rank_cycle
```

### 4.3 Edge reconstruction checks

For every transport relation with defined source and target bases:

1. reconstruct `C_(target<-source)`
2. recompute its singular values
3. compare them with stored `overlap_singular_values`
4. recompute the Gate12A polar factor or active partial isometry under the frozen Gate12A law
5. compare it with stored `transport_matrix_local`

Required run-level diagnostics include:

```text
overlap_singular_value_max_abs_error
transport_reconstruction_max_fro_error
reconstructed_edge_count
failed_edge_reconstruction_count
```

The exact tolerances must be explicit CLI or module constants and recorded in the manifest.

### 4.4 Cycle census

The preflight must report counts for:

```text
total_gate12a_residual_bearing_explicit_triangle_count
defined_equal_rank_triangle_count
common_rank_1_triangle_count
common_rank_2_triangle_count
common_rank_3_triangle_count
common_rank_ge_4_triangle_count
```

The current implementation may expose `r_max = 3`, but Gate12C must derive the actual maximum from artifacts rather than hard-code it as doctrine.

### 4.5 Nontrivial compression ranks

For each eligible equal-rank cycle with common rank `r`, enumerate:

```text
q in {1, ..., r - 1}
```

No `q` exists for `r = 1`; those cycles are counted but are not eligible for a nontrivial associator probe.

### 4.6 All cyclic roots

For every eligible cycle, evaluate all three cyclic root choices.

The canonical `base_node_id` remains artifact metadata only. It must not be the sole evaluation root.

### 4.7 Stable-cut census

For every cycle, root, and nontrivial `q`, evaluate the two intermediate products required by the alpha composition law.

For a matrix `M` with singular values in descending order, define the relative split gap at rank `q`:

```text
gap_q(M) = (sigma_q - sigma_(q+1)) / max(sigma_1, epsilon)
```

A cut is stable only if:

```text
gap_q(M) > tau_split_rel
```

The preflight must report:

```text
probe_configuration_count
stable_both_inner_cut_count
near_degenerate_left_cut_count
near_degenerate_right_cut_count
near_degenerate_both_cut_count
eligible_cycle_count_with_at_least_one_stable_q
```

A near-degenerate cut is non-promotable. The implementation must not select an arbitrary SVD basis across an unresolved split and then report that result as a stable associator.

### 4.8 Ordinary associativity null

For every eligible cycle and every cyclic root, reconstruct the ordered overlaps `M0`, `M1`, `M2` and verify:

```text
(M2 @ M1) @ M0 ~= M2 @ (M1 @ M0)
```

The preflight must emit:

```text
ordinary_associator_max_fro
ordinary_associator_failed_count
```

Failure above the declared numerical tolerance is a preflight failure, not an empirical result.

### 4.9 Gate12C-0 outputs

Required first-pass files:

```text
manifest.json
gate12c_feasibility_preflight.json
gate12c_feasibility_cycle_census.csv
gate12c_feasibility_cut_census.jsonl
gate12c_feasibility_read.md
checksums.json
```

The preflight does not emit Type-III candidates.

## 5. Gate12C-1 Equal-Rank Alpha Composition Law

### 5.1 Ordered triangle

Let one Gate12A-defined residual-bearing explicit triangle, after selecting a cyclic root, be:

```text
v0 -> v1 -> v2 -> v0
```

Define:

```text
M0 = C_(v1<-v0)
M1 = C_(v2<-v1)
M2 = C_(v0<-v2)
```

All three matrices are square `r x r` because alpha is equal-rank only.

The ordinary product is:

```text
H_C = M2 @ M1 @ M0
```

### 5.2 Frozen compression operator

Define `Q_q(M)` as the best rank-`q` approximation under compact SVD:

```text
M = U Sigma V^T
Q_q(M) = U[:, :q] Sigma[:q] V[:, :q]^T
```

`Q_q` is defined for alpha only when the declared relative spectral split at `q` is stable.

No Procrustes alignment, gauge fixing, learned map, semantic correction, or model-specific tuning is part of `Q_q`.

### 5.3 Parenthesized compressed overlap compositions

Define:

```text
L_q = Q_q(M2 @ M1) @ M0
R_q = M2 @ Q_q(M1 @ M0)
A_q = L_q - R_q
```

The output names `L_q` and `R_q` refer to compressed raw-overlap compositions, not Gate12A polar holonomies.

### 5.4 Primary readouts

Required operator and scalar readouts are:

```text
compressed_overlap_left_operator = L_q
compressed_overlap_right_operator = R_q
compressed_overlap_associator_operator = A_q
compressed_overlap_associator_fro = ||A_q||_F
```

A scale-aware relative readout may be emitted as:

```text
compressed_overlap_associator_rel =
  ||L_q - R_q||_F /
  (sqrt(2 * (||L_q||_F^2 + ||R_q||_F^2)) + epsilon)
```

### 5.5 Closure naming boundary

The following Gate12C names are required:

```text
compressed_overlap_closure_left_fro  = ||L_q - I_r||_F
compressed_overlap_closure_right_fro = ||R_q - I_r||_F
compressed_overlap_closure_gap_abs   = abs(left - right)
```

The following existing Gate12A/Gate12B field remains separate and unchanged:

```text
gate12a_holonomy_residual_fro
```

Gate12C must not emit raw-overlap closure fields under the generic names:

```text
left_closure_fro
right_closure_fro
holonomy_left
holonomy_right
```

without the `compressed_overlap_` qualifier.

### 5.6 Cycle-root aggregation

For each cycle and `q`, emit all three root-level rows.

A cycle-level RMS may be emitted:

```text
compressed_overlap_associator_root_rms =
  sqrt(mean_root(compressed_overlap_associator_fro^2))
```

A root spread may be emitted:

```text
compressed_overlap_associator_root_spread =
  max_root(value) - min_root(value)
```

The root-level rows remain primary. A cycle aggregate must not erase observer/root sensitivity.

## 6. Required Nulls and Stability Controls

### 6.1 Ordinary-product null

Required:

```text
ordinary_associator_fro = ||(M2 M1) M0 - M2 (M1 M0)||_F
```

This is an implementation and ordering check.

### 6.2 No-compression null

Run the same code path with:

```text
q = r
```

and require the resulting associator to remain within the declared numerical tolerance.

This is a compression-code regression check, not independent empirical evidence.

### 6.3 Gauge covariance check

Apply deterministic node-wise orthogonal signed permutations:

```text
U_v' = U_v G_v
C_(target<-source)' = G_target^T C_(target<-source) G_source
```

Recompute the scalar associator readouts.

The scalar Frobenius readouts must remain stable within a declared tolerance whenever the relevant SVD cuts remain admissible.

The alpha contract uses gauge covariance and gauge-stability testing. It does not insert a gauge-fixing map into the core composition law.

### 6.4 Spectrum-preserving orientation null

For a promoted empirical analysis, compare observed values with a deterministic-seeded null that preserves each edge singular spectrum while randomizing left and right orientation:

```text
C_e^(b) = L_e^(b) Sigma_e R_e^(b)^T
```

The null must preserve at least:

- edge matrix shape
- edge rank
- edge singular values
- common cycle rank
- compression rank `q`
- triangle relation metadata

The exact draw count and seed must be declared in the manifest.

Raw positivity of the associator is not sufficient. A promoted excess claim requires comparison with the matched null.

### 6.5 Distinctness from existing defects

Any later analysis must retain separate fields for:

```text
pairwise edge defect
Gate12A polar holonomy residual
Gate12C compressed-overlap associator
```

Gate12C must not collapse those axes into one total defect score in alpha.

## 7. Gate12C-1 Artifact Contract

Suggested output directory:

```text
runs/gate12c_compressed_overlap_associator_<run_id>/
```

Required files:

```text
manifest.json
triangle_associator_registry.jsonl
triangle_associator_arrays.npz
cycle_associator_summary.jsonl
compression_sweep_summary.csv
gauge_stability_summary.json
spectral_orientation_null_summary.jsonl
gate12c_status.json
gate12c_read.md
checksums.json
```

### 7.1 Array artifact

`triangle_associator_arrays.npz` should carry:

```text
compressed_overlap_left_operator
compressed_overlap_right_operator
compressed_overlap_associator_operator
```

with deterministic row indices declared in the registry.

### 7.2 Minimum registry fields

Each root/q probe row must include:

```text
probe_id
cycle_id
canonical_base_node_id
evaluation_root_node_id
root_rotation_index
ordered_node_id_path
ordered_edge_id_path
ordered_relation_kind_path
cycle_rank
compression_rank_q
left_inner_split_gap_rel
right_inner_split_gap_rel
left_cut_status
right_cut_status
truncation_status
ordinary_associator_fro
no_compression_associator_fro
compressed_overlap_associator_fro
compressed_overlap_associator_rel
compressed_overlap_closure_left_fro
compressed_overlap_closure_right_fro
compressed_overlap_closure_gap_abs
gate12a_holonomy_residual_fro
edge_compatibility_gap_max
operator_array_index
promotable
```

Suggested `truncation_status` values:

```text
stable_both_active
near_degenerate_left
near_degenerate_right
near_degenerate_both
compression_inactive
undefined_input
```

## 8. Promotion Boundary

A nonzero `compressed_overlap_associator_fro` is not by itself a Type-III result.

A probe row is structurally promotable only if:

- it belongs to the exact Gate12A residual-bearing explicit triangle surface
- the cycle is Gate12A-defined equal-rank with common rank `r >= 2`
- ordinary associativity null passes
- no-compression null passes
- both required rank cuts are spectrally admissible
- gauge-stability check passes
- reconstructed overlaps agree with stored Gate12A spectra and transports

An empirical excess claim additionally requires:

- matched spectrum-preserving orientation null comparison
- explicit separation from pairwise edge defect
- explicit separation from Gate12A holonomy residual
- block-aware aggregation at the source-run or source-sample level rather than treating overlapping cycles as independent observations

The phrase `Type-III defect` must remain internal shorthand until those controls pass. The external alpha name is:

> compressed-overlap parenthesization sensitivity

## 9. Gate12B Overlay Boundary

Gate12C core must run on the full eligible Gate12A-defined equal-rank surface before any Gate12B overlay.

A later, separate reader may join by `cycle_id` to compare:

```text
residual_chord=3
```

against:

```text
residual_chord=1|trusted_tree=2
```

or to inspect Gate12B flat/high-tension surfaces.

That overlay is post-hoc analysis. It must not influence Gate12C computation, compression rank selection, spectral-cut admissibility, null construction, or promotion.

## 10. Explicit Deferrals

The following are outside Gate12C-1 alpha:

- rectangular rank-mismatch overlap associators
- partial-isometry associator laws
- arbitrary cycle enumeration
- four-node open-chain registries
- higher-category or gerbe language
- octonionic coefficient systems
- Cayley-Dickson constructions
- physical nonassociativity claims
- gravity, twistor, topos, or field-theory projection claims
- phenotype classification
- answer correctness classification
- model-quality ranking
- weight-level causal interpretation

## 11. Implementation Files

The first implementation chat should add, in order:

```text
tools/inspect_gate12c_associator_feasibility.py
tools/test_inspect_gate12c_associator_feasibility.py
```

Only after Gate12C-0 passes on a declared real artifact set should it add:

```text
tools/run_gate12c_compressed_overlap_associator.py
tools/test_run_gate12c_compressed_overlap_associator.py
docs/gate12c_compressed_overlap_associator_runbook.md
```

Existing Gate12A and Gate12B semantics must remain unchanged.

## 12. Required Regression Tests

Gate12C-0 tests must include:

- missing required artifact rejection
- source output-directory alias rejection
- source nested output-directory rejection
- overlap reconstruction from `basis_factor`
- stored singular-spectrum consistency
- stored polar-transport consistency
- common-rank census
- rank-1 nontrivial-probe exclusion
- all three cyclic roots counted
- lexicographically sorted `edge_id_path` not used as traversal order
- ordinary matrix associativity null
- stable split-gap recognition
- near-degenerate split rejection
- deterministic manifest and checksum output

Gate12C-1 tests must include:

- `q = r` no-compression null
- deterministic positive compressed associator fixture
- near-degenerate cut remains non-promotable
- deterministic node signed-permutation gauge stability
- required `compressed_overlap_` closure field names
- separate preservation of `gate12a_holonomy_residual_fro`
- all three cyclic roots emitted
- source Gate12A directory remains read-only
- deterministic array row indexing

## 13. Kill Switches

Stop before Gate12C-1 if any of the following holds:

- required node or transport arrays are absent
- overlap singular spectra cannot be reconstructed within tolerance
- polar transport cannot be reconstructed within tolerance
- ordinary associativity null fails
- eligible equal-rank `r >= 2` cycle count is below the declared minimum
- stable nontrivial cuts are absent or below the declared minimum

Do not promote an empirical Gate12C result if:

- gauge stability fails
- observed associator is not above the matched spectrum-preserving null
- the readout is explained almost entirely by pairwise edge compatibility loss
- the readout is only a monotone restatement of Gate12A holonomy residual
- the result appears only after selecting Gate12B candidates first

## 14. Implementation Handoff Sentence

The new implementation chat should begin from this exact boundary:

> Implement Gate12C-0 first as a read-only feasibility preflight over Gate12A-defined residual-bearing explicit triangles. Empirical feasibility is currently unknown. Gate12C-1 alpha is equal-rank only, requires common rank `r >= 2`, uses reconstructed raw overlaps, keeps compressed-overlap closure distinct from Gate12A polar holonomy closure, and defers rectangular partial-isometry associators.

## 15. Short Sentence

Gate12C does not yet claim nonassociative physics. It first counts whether the existing Gate12A residual-bearing explicit-triangle surface can support a stable, equal-rank, low-rank recomposition audit at all.
