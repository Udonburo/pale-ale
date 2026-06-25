# Gate12C Equal-Rank Alpha Implementation Contract

Status: implementation contract
Role: bounded contract for implementing Gate12C-1 equal-rank compressed-overlap parenthesization sensitivity on the Gate12C-0 admissible surface, not a rectangular-rank associator contract, not a Gate12B overlay, and not a physical nonassociativity claim
Date: 2026-06-25

This memo proceeds from:

- `231_GATE12C_ASSOCIATOR_FEASIBILITY_AND_EQUAL_RANK_ALPHA_CONTRACT.md`
- `232_GATE12C_FIRST_REAL_ARTIFACT_FEASIBILITY_CENSUS_PLAN.md`
- `233_GATE12C_FIRST_REAL_ARTIFACT_FEASIBILITY_CENSUS_MEMO.md`
- `tools/inspect_gate12c_associator_feasibility.py`

## 0. Executive Decision

Gate12C-0 found a broad real-artifact equal-rank surface:

```text
12 / 12 canonical cases contract pass
12 / 12 canonical cases empirical_surface_status = pass_declared_minimum
4,560 rank-3 eligible cycles
27,360 / 27,360 stable-both nontrivial SVD cuts
0 near-degenerate cuts
0 reconstruction failures
0 ordinary associativity-null failures
```

This authorizes drafting Gate12C-1 alpha implementation scope.

It does not authorize a Type-III claim, a physical nonassociativity claim, rectangular rank-mismatch support, or Gate12B-selected filtering.

The Gate12C-1 alpha question is:

> On Gate12A-defined residual-bearing explicit triangles that are already contract-feasible and equal-rank, does the frozen low-rank recomposition law over reconstructed raw overlaps produce measurable parenthesization sensitivity after all declared null and stability controls pass?

## 1. Scope Boundary

Gate12C-1 alpha must operate only on:

- Gate12A-defined residual-bearing explicit triangles
- `holonomy_status == defined`
- every leg has `transport_case == equal_rank_orthogonal`
- all three nodes share one common positive rank `r`
- `r >= 2`
- both required nontrivial SVD cuts are stable at the selected `q`

The first implementation should be general for equal-rank `r >= 2`, but the canonical real-artifact surface from `233` is currently:

```text
cycle_rank = 3
q in {1, 2}
all three cyclic roots evaluated
```

Gate12C-1 must not:

- implement rectangular rank-mismatch associators
- implement partial-isometry associator laws
- infer or regenerate model outputs
- modify Gate12A or Gate12B artifacts
- consume Gate12B high/flat candidates in core computation
- tune thresholds after seeing associator values
- call stable-cut availability associator evidence
- call a positive associator row Type-III evidence

## 2. Required Inputs

The runner must accept one Gate12A artifact directory and one output directory.

Required Gate12A files:

```text
manifest.json
node_local_object_registry.jsonl
node_local_object_arrays.npz
transport_relation_registry.jsonl
transport_operator_arrays.npz
explicit_triangle_cycle_registry.jsonl
triangle_holonomy_registry.jsonl
```

Required arrays:

```text
node_local_object_arrays.npz
  basis_factor
  rank_active

transport_operator_arrays.npz
  transport_matrix_local
  overlap_singular_values
  active_rank
```

Gate12C-1 should reuse the Gate12C-0 reconstruction discipline:

1. Reconstruct every raw overlap from active bases:

```text
C_(target<-source) = U_target.T @ U_source
```

2. Recompute and compare stored singular spectra.
3. Recompute and compare the Gate12A frozen polar or partial-isometry transport law.
4. Reconstruct ordered triangle edges from consecutive `node_id_path` pairs and transport registry source/target ids.
5. Reject source/output aliasing and nested output directories.
6. Verify the Gate12A source directory remains byte-for-byte unchanged over the required input files.

## 3. Frozen Alpha Law

For a chosen cyclic root:

```text
v0 -> v1 -> v2 -> v0
M0 = C_(v1<-v0)
M1 = C_(v2<-v1)
M2 = C_(v0<-v2)
```

All matrices are square `r x r`.

For each nontrivial compression rank:

```text
q in {1, ..., r - 1}
```

define the frozen rank-`q` approximation:

```text
M = U Sigma V.T
Q_q(M) = U[:, :q] Sigma[:q] V[:, :q].T
```

The parenthesized compressed-overlap compositions are:

```text
L_q = Q_q(M2 @ M1) @ M0
R_q = M2 @ Q_q(M1 @ M0)
A_q = L_q - R_q
```

The primary scalar readout is:

```text
compressed_overlap_associator_fro = ||A_q||_F
```

The relative scalar readout is:

```text
compressed_overlap_associator_rel =
  ||L_q - R_q||_F /
  (sqrt(2 * (||L_q||_F^2 + ||R_q||_F^2)) + epsilon)
```

Gate12C-1 must use raw overlaps for this law, not Gate12A polar transports.

## 4. Stability Gate

For each candidate root and `q`, compute singular values of both inner products:

```text
left_inner = M2 @ M1
right_inner = M1 @ M0
```

The split gap at `q` is:

```text
gap_q(M) = (sigma_q - sigma_(q+1)) / max(sigma_1, epsilon)
```

A truncation is active only when:

```text
left_gap_q > tau_split_rel
right_gap_q > tau_split_rel
```

Near-degenerate cuts must remain aggregation-ineligible. The implementation must not preserve or choose an arbitrary SVD basis across an unresolved split and report that as an active alpha result.

Suggested `truncation_status` values:

```text
stable_both_active
near_degenerate_left
near_degenerate_right
near_degenerate_both
compression_inactive
undefined_input
```

## 5. Required Nulls and Controls

### 5.1 Ordinary Associativity Null

For every eligible root:

```text
ordinary_associator_fro = ||(M2 @ M1) @ M0 - M2 @ (M1 @ M0)||_F
```

Failure above tolerance is an implementation or ordering failure, not an empirical Gate12C result.

### 5.2 No-Compression Null

Run the same code path with:

```text
q = r
```

The no-compression associator must remain within the declared numerical tolerance.

This is a regression check for the compression code path. It is not independent empirical evidence.

### 5.3 Gauge Covariance

Apply deterministic node-wise orthogonal signed permutations:

```text
U_v' = U_v G_v
C_(target<-source)' = G_target.T @ C_(target<-source) @ G_source
```

Recompute the full compressed-overlap operators and scalar readouts.

At the selected root `v0`, the associator operator must transform covariantly:

```text
A_q_transformed ~= G_0.T @ A_q @ G_0
```

Required gauge fields:

```text
gauge_operator_covariance_fro = ||A_q_transformed - G_0.T @ A_q @ G_0||_F
gauge_scalar_delta_abs = abs(||A_q_transformed||_F - ||A_q||_F)
gauge_cut_status_preserved
gauge_scalar_status
```

Gauge pass requires both:

- operator covariance within the declared tolerance
- scalar Frobenius stability within the declared tolerance

For rows where the gauge-transformed cut is no longer stable, set `gauge_cut_status_preserved = false` and do not treat the row as aggregation-eligible.

The implementation must not insert a gauge-fixing map into the core law.

### 5.4 Spectrum-Preserving Orientation Null

The orientation null is an edge-wise coherence-destroying operator-level null. It preserves edge singular spectra, but it is not a gauge transform and is not required to be realizable by one globally consistent set of node bases.

For each cycle and draw:

- generate one randomized triple of edge overlap operators
- preserve each edge singular spectrum
- reuse that same randomized triangle across all three cyclic roots and every `q`
- do not generate independent null triangles per root/q row

For each randomized edge:

- edge matrix shape
- edge rank
- edge singular values
- common cycle rank
- triangle relation metadata

The null edge operator is:

```text
C_e^(b) = L_e^(b) Sigma_e R_e^(b).T
```

where `Sigma_e` is the observed edge singular spectrum, and `L_e^(b)` and `R_e^(b)` are deterministic seeded orthogonal orientations.

Seed derivation must be independent of registry row order and execution parallelism. For each orientation matrix, derive the seed from a stable hash of:

```text
SHA256(
  orientation_null_seed
  cycle_id
  edge_id
  draw_index
  left_or_right_orientation_label
)
```

The deterministic orthogonal generator is frozen as:

1. Fill an `r x r` matrix `Z` row-major from SHA256 counter blocks keyed by the orientation seed.
2. Convert hash-derived uniform pairs to standard-normal entries with a Box-Muller transform.
3. Compute `Z = Q R` by QR decomposition.
4. Sign-normalize `Q` by multiplying each column by `sign(diag(R))`, with zero signs replaced by `+1`.
5. Use the resulting `Q` as the orientation matrix.

All implementation manifests must record the generator id, seed, requested draw count, and max attempt count.

Required null settings must be declared in the manifest:

```text
orientation_null_seed
orientation_null_requested_draw_count
orientation_null_max_attempt_count
orientation_null_mode
orientation_null_orthogonal_generator
```

For every null cycle draw and root/q:

1. compute the same left and right split gaps
2. mark the row-draw `invalid_cut` when either cut is near-degenerate
3. do not force an arbitrary SVD basis through an invalid cut

Generate attempts until the requested valid draw count is reached or `orientation_null_max_attempt_count` is exhausted.

Required per-row null accounting fields:

```text
orientation_null_requested_draw_count
orientation_null_valid_draw_count
orientation_null_invalid_cut_count
orientation_null_attempt_count
orientation_null_status
```

If valid draws are insufficient:

```text
orientation_null_status = insufficient_valid_draws
aggregation_eligible = false
```

For observed scalar `a_obs` and valid null values `a_b`, define:

```text
orientation_null_empirical_p_upper =
  (1 + count(a_b >= a_obs)) / (1 + B_valid)

orientation_null_robust_z =
  (a_obs - median(a_b)) /
  (1.4826 * MAD(a_b) + epsilon)
```

Required null summary fields:

```text
orientation_null_median
orientation_null_mad
orientation_null_mean
orientation_null_std
orientation_null_empirical_p_upper
orientation_null_robust_z
orientation_null_scale_degenerate
```

Row-level p-values and robust z-scores are descriptive telemetry only. No row-level discovery claim is authorized by this contract. Multiple-testing boundaries and block-aware claim aggregation require a separate empirical execution plan.

Do not freeze a scientific excess threshold in the implementation contract unless a later plan explicitly justifies and predeclares it. Gate12C-1 implementation should emit the statistics; a later empirical plan decides claim thresholds and aggregation.

Raw positivity is not sufficient for any empirical excess claim.

## 6. Required Outputs

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

Array artifact:

```text
triangle_associator_arrays.npz
  compressed_overlap_left_operator
  compressed_overlap_right_operator
  compressed_overlap_associator_operator
```

All JSON outputs must reject NaN and infinity. Output ordering, row indexing, manifests, and checksums must be deterministic.

## 7. Minimum Registry Fields

Each root/q row must include:

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
source_sample_block_id
source_block_status
measurement_status
control_status
aggregation_eligible
gauge_operator_covariance_fro
gauge_scalar_delta_abs
gauge_cut_status_preserved
gauge_scalar_status
orientation_null_status
orientation_null_excess_status
orientation_null_requested_draw_count
orientation_null_valid_draw_count
orientation_null_invalid_cut_count
orientation_null_attempt_count
orientation_null_median
orientation_null_mad
orientation_null_mean
orientation_null_std
orientation_null_empirical_p_upper
orientation_null_robust_z
orientation_null_scale_degenerate
operator_array_index
```

The `compressed_overlap_` prefix is mandatory for raw-overlap closure readouts. Do not emit ambiguous fields such as `left_closure_fro`, `right_closure_fro`, `holonomy_left`, or `holonomy_right`.

Required row statuses:

```text
measurement_status
control_status
aggregation_eligible
orientation_null_status
orientation_null_excess_status
```

Do not emit `promotable`. If an older draft or helper uses that word, rename it to `aggregation_eligible` and apply the narrower definition in Section 9.

`source_sample_block_id` and `source_block_status` are required for later block-aware aggregation.

When all cycle nodes share one `sample_XXXXXX` prefix, set:

```text
source_sample_block_id = sample_XXXXXX
source_block_status = single_sample
```

Otherwise set:

```text
source_sample_block_id = mixed_or_undefined
source_block_status = mixed_or_undefined
```

Later empirical claims must aggregate by source-sample or source-run block and must not treat overlapping cycle rows as independent samples.

## 8. Cycle-Level Summaries

Cycle summaries may aggregate root/q rows but must not erase root sensitivity.

Suggested cycle-level fields:

```text
cycle_id
cycle_rank
eligible_root_q_count
stable_both_active_count
compressed_overlap_associator_root_rms
compressed_overlap_associator_root_max
compressed_overlap_associator_root_spread
ordinary_associator_max_fro
no_compression_associator_max_fro
gauge_stable_row_count
orientation_null_complete_row_count
aggregation_eligible_row_count
```

The root-level registry remains primary.

## 9. Status and Aggregation Boundary

Gate12C-1 separates measurement, control completion, aggregation eligibility, and null excess.

`measurement_status` describes whether the row produced the requested compressed-overlap operators and scalar readouts.

`control_status` describes whether reconstruction, ordinary null, no-compression null, stable cuts, gauge checks, and orientation-null draw completion passed.

`measurement_status` values:

```text
not_evaluated
measured
invalid_input
```

`control_status` values:

```text
not_evaluated
pass
fail
incomplete
```

`aggregation_eligible = true` only when all of the following hold:

- exact Gate12A residual-bearing explicit triangle surface
- equal-rank common-rank cycle with `r >= 2`
- `ordinary_associator_fro` within tolerance
- `no_compression_associator_fro` within tolerance
- both nontrivial rank cuts stable
- reconstructed overlaps agree with stored Gate12A spectra
- reconstructed Gate12A transports agree with stored transports
- gauge operator covariance passes
- gauge scalar stability passes
- the requested number of valid orientation-null draws has completed

`aggregation_eligible = true` does not mean the observed associator exceeds the null. It does not authorize a Type-III claim.

`orientation_null_status` records draw completion:

```text
not_evaluated
complete
insufficient_valid_draws
invalid_input
```

`orientation_null_excess_status` records only descriptive comparison status, for example:

```text
not_evaluated
descriptive_only
scale_degenerate
```

An empirical excess claim additionally requires:

- observed scalar readout exceeds the matched null under declared criteria
- separation from pairwise edge compatibility loss
- separation from Gate12A polar holonomy residual
- block-aware aggregation by source run or source sample

Do not label any output row as Type-III.

Do not include a generic `promotable` field; it conflates control completion with empirical excess. Use `aggregation_eligible` and `orientation_null_excess_status` separately.

## 10. CLI and Process Semantics

The command-line process status reports whether the contract execution succeeded, not whether an empirical excess was observed.

A valid run with no orientation-null excess must exit `0`.

These data outcomes must not force a nonzero process exit when the run otherwise emits complete deterministic artifacts:

- no empirical null excess
- zero aggregation-eligible rows
- rows marked `orientation_null_status = insufficient_valid_draws` with requested, valid, invalid, and attempt counts recorded
- rows marked `orientation_null_excess_status = descriptive_only`

Exit `1` is reserved for contract or implementation failures, including:

- missing required input artifacts
- source/output aliasing or nested output paths
- source artifact mutation
- raw-overlap reconstruction mismatch
- stored singular-spectrum or transport reconstruction mismatch
- ordinary associativity-null failure
- no-compression null failure
- gauge operator covariance failure
- gauge scalar stability failure
- nondeterministic registry, array, manifest, or checksum output
- invalid JSON numeric output such as NaN or infinity

`gate12c_status.json` must distinguish process success from empirical excess status.

## 11. Gate12B Overlay Boundary

Gate12C-1 core computation must run over the full eligible Gate12A-defined equal-rank surface before any Gate12B overlay.

A later reader may join by `cycle_id` to compare relation signatures or Gate12B observer-relative bands. That overlay must not influence:

- input cycle selection
- compression rank selection
- spectral-cut admissibility
- gauge transforms
- null construction
- aggregation eligibility or orientation-null excess status

## 12. Required Tests

Add focused regression coverage for:

- required Gate12A artifact rejection
- source/output alias and nested-output rejection
- source byte-for-byte immutability
- raw overlap reconstruction from `basis_factor`
- stored singular-spectrum and transport reconstruction consistency
- ordered edge reconstruction from `node_id_path`, not `edge_id_path`
- equal-rank-only eligibility
- all three cyclic roots emitted
- `q in {1, ..., r - 1}` enumeration
- `q = r` no-compression null
- ordinary associativity null
- stable split-gap activation
- near-degenerate split non-aggregation-eligibility
- deterministic positive compressed associator fixture
- required `compressed_overlap_` field names
- preservation of `gate12a_holonomy_residual_fro`
- required status fields and absence of a generic `promotable` field
- source-sample block provenance, including `mixed_or_undefined`
- gauge operator covariance under deterministic signed permutations
- gauge scalar stability under deterministic signed permutations
- gauge cut-status preservation accounting
- matched spectrum-preserving orientation null determinism
- one randomized null triangle reused across all roots and `q`
- orientation-null seed independence from row order and parallel execution
- frozen SHA256 plus QR sign-normalized orthogonal generator
- null split-gap invalid-cut detection without arbitrary SVD basis selection
- valid-draw retry, max-attempt exhaustion, and insufficient-valid-draw accounting
- orientation-null p-value, robust-z, and scale-degenerate formulas
- descriptive-only row-level p/z status
- valid no-excess run exits `0`
- contract and implementation failures exit `1`
- deterministic registry and array row indexing
- deterministic manifest and checksum output

Existing Gate12A and Gate12B regression tests must still pass.

## 13. Implementation Files

The implementation should add:

```text
tools/run_gate12c_compressed_overlap_associator.py
tools/test_run_gate12c_compressed_overlap_associator.py
docs/gate12c_compressed_overlap_associator_runbook.md
```

Do not modify Gate12A or Gate12B implementation files unless a genuine blocking incompatibility is found. If the current code contradicts this contract or `231`, stop and report the discrepancy before editing the canonical contract.

## 14. Validation

Before opening an implementation PR, run:

```text
python -m unittest tools.test_run_gate12c_compressed_overlap_associator
python -m unittest tools.test_inspect_gate12c_associator_feasibility
python -m unittest tools.test_run_gate12a_discrete_connection_audit tools.test_run_gate12b_observer_relative_coarse_grained_closure
git diff --check
```

Do not stage generated `runs/` directories.

## 15. Short Sentence

Gate12C-1 may now measure compressed-overlap parenthesization sensitivity on the broad equal-rank `r = 3` surface found by Gate12C-0, but only with the declared nulls, gauge checks, orientation null, and claim boundary in place.
