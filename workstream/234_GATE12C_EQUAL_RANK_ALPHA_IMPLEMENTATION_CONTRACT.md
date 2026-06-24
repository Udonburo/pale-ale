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

Near-degenerate cuts must remain non-promotable. The implementation must not preserve or choose an arbitrary SVD basis across an unresolved split and report that as an active alpha result.

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

Recompute scalar readouts.

For admissible cuts that remain stable under the transformed basis, the Frobenius scalar readouts must remain stable within tolerance. The implementation must not insert a gauge-fixing map into the core law.

### 5.4 Spectrum-Preserving Orientation Null

For each observed row, construct a deterministic-seeded null that preserves:

- edge matrix shape
- edge rank
- edge singular values
- common cycle rank
- compression rank `q`
- triangle relation metadata

The null draw should use:

```text
C_e^(b) = L_e^(b) Sigma_e R_e^(b).T
```

where `L_e^(b)` and `R_e^(b)` are deterministic seeded orthogonal orientations.

Required null settings must be declared in the manifest:

```text
orientation_null_seed
orientation_null_draw_count
orientation_null_mode
```

A promoted empirical excess claim requires comparison with this matched null. Raw positivity is not sufficient.

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
gauge_scalar_status
orientation_null_status
orientation_null_draw_count
orientation_null_excess_z
operator_array_index
promotable
```

The `compressed_overlap_` prefix is mandatory for raw-overlap closure readouts. Do not emit ambiguous fields such as `left_closure_fro`, `right_closure_fro`, `holonomy_left`, or `holonomy_right`.

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
orientation_null_promotable_row_count
promotable_row_count
```

The root-level registry remains primary.

## 9. Promotion Boundary

A row is structurally promotable only when all of the following hold:

- exact Gate12A residual-bearing explicit triangle surface
- equal-rank common-rank cycle with `r >= 2`
- `ordinary_associator_fro` within tolerance
- `no_compression_associator_fro` within tolerance
- both nontrivial rank cuts stable
- reconstructed overlaps agree with stored Gate12A spectra
- reconstructed Gate12A transports agree with stored transports
- gauge scalar check passes
- matched spectrum-preserving orientation null is evaluated

An empirical excess claim additionally requires:

- observed scalar readout exceeds the matched null under declared criteria
- separation from pairwise edge compatibility loss
- separation from Gate12A polar holonomy residual
- block-aware aggregation by source run or source sample

Do not label any output row as Type-III.

## 10. Gate12B Overlay Boundary

Gate12C-1 core computation must run over the full eligible Gate12A-defined equal-rank surface before any Gate12B overlay.

A later reader may join by `cycle_id` to compare relation signatures or Gate12B observer-relative bands. That overlay must not influence:

- input cycle selection
- compression rank selection
- spectral-cut admissibility
- gauge transforms
- null construction
- promotion status

## 11. Required Tests

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
- near-degenerate split non-promotion
- deterministic positive compressed associator fixture
- required `compressed_overlap_` field names
- preservation of `gate12a_holonomy_residual_fro`
- gauge scalar stability under deterministic signed permutations
- matched spectrum-preserving orientation null determinism
- deterministic registry and array row indexing
- deterministic manifest and checksum output

Existing Gate12A and Gate12B regression tests must still pass.

## 12. Implementation Files

The implementation should add:

```text
tools/run_gate12c_compressed_overlap_associator.py
tools/test_run_gate12c_compressed_overlap_associator.py
docs/gate12c_compressed_overlap_associator_runbook.md
```

Do not modify Gate12A or Gate12B implementation files unless a genuine blocking incompatibility is found. If the current code contradicts this contract or `231`, stop and report the discrepancy before editing the canonical contract.

## 13. Validation

Before opening an implementation PR, run:

```text
python -m unittest tools.test_run_gate12c_compressed_overlap_associator
python -m unittest tools.test_inspect_gate12c_associator_feasibility
python -m unittest tools.test_run_gate12a_discrete_connection_audit tools.test_run_gate12b_observer_relative_coarse_grained_closure
git diff --check
```

Do not stage generated `runs/` directories.

## 14. Short Sentence

Gate12C-1 may now measure compressed-overlap parenthesization sensitivity on the broad equal-rank `r = 3` surface found by Gate12C-0, but only with the declared nulls, gauge checks, orientation null, and claim boundary in place.
