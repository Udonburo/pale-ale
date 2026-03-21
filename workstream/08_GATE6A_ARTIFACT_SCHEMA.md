# Gate6-A Artifact Schema

Status: Draft
Role: Tracked RFC / artifact contract for native local span outputs
Date: 2026-03-17

## 0. Schema Goal

Gate6-A must emit:

1. the native local object
2. a compatibility view for benchmarking
3. deterministic manifests and provenance

The artifact family must support:

- downstream transport experiments
- reruns and diffing
- rank, gauge, and degeneracy analysis
- benchmark comparison with Gate5

This is not a CSV-only system.
Native local objects are too rich for flat scalar tables alone.

## 1. Run Directory Layout

Recommended layout:

`runs/gate6_native_local_span_<run_id>/`

- `manifest.json`
- `step_index.jsonl`
- `native_object_arrays.npz`
- `compatibility_input.json`
- `aggregate_summary.md`
- `checksums.json`

## 2. manifest.json

Required fields:

- `run_id`
- `schema_version`
- `method_id = "native_local_span_gate6a_v1"`
- `source_model_id`
- `source_model_revision`
- `source_layer_id`
- `input_source_path`
- `input_sha256`
- `code_git_commit`
- `builder_script_sha256`
- `tau_norm_abs`
- `tau_rank_abs`
- `tau_rank_rel`
- `tau_sign_tie_abs`
- `normalization_mode`
- `construction_mode = "anchor_plus_relations_svd_v1"`
- `sign_fix_mode = "largest_abs_component_positive_first_index_tie_v1"`
- `compatibility_embedding = "local_rank_to_local8_zero_pad_v1"`
- `n_samples_total`
- `n_token_steps_total`

## 3. step_index.jsonl

One JSON object per token step.

Required keys:

- `sample_id`
- `step`
- `token_text`
- `label_token`
- `baseline_logprob`
- `baseline_entropy`
- `offset_start`
- `offset_end`
- `array_row_index`
- `rank_local`
- `flags_compact`

This file is the searchable row index.
It should remain human-readable.

## 4. native_object_arrays.npz

This is the main binary artifact.

Required arrays:

### A. basis

Shape:

- `[N, d_model, 3]`

Definition:

- sign-fixed orthonormal basis columns
- unused trailing columns zero-padded if rank `< 3`

### B. projector_factor

Shape:

- may reuse `basis`

Policy:

- the canonical projector is represented implicitly as `basis @ basis.T`
- do not store dense full `P_t in R^(d x d)` in v1

### C. coords_local

Shape:

- `[N, 3, 3]`

Definition:

- local coordinates of `(V, Splus, Sminus)`
- padded rows if rank `< 3`

Convention:

- axis 1 = local coordinate axis
- axis 2 = observable index order `(V, Splus, Sminus)`

### D. gram_raw

Shape:

- `[N, 3, 3]`

Definition:

- Gram matrix of normalized raw observables `(v, p, m)`

### E. singular_values

Shape:

- `[N, 3]`

### F. rank_local

Shape:

- `[N]`

### G. norms_raw

Shape:

- `[N, 3]`

Order:

- `(norm_V_raw, norm_Splus_raw, norm_Sminus_raw)`

### H. flags

Shape:

- `[N, K]` or encoded integer mask

Recommended fields:

- `all_finite`
- `used_sign_fix`
- `rank_drop_to_2`
- `rank_drop_to_1`
- `near_degenerate`

### I. compat_local8

Shape:

- `[N, 3, 8]`

Definition:

- local coordinate compatibility embedding for `(V_local8, Splus_local8, Sminus_local8)`

This array is what allows Gate5-style transport scoring to be rerun under the new boundary.

## 5. compatibility_input.json

This is a transport-ready compatibility artifact.

Goal:

- let existing Gate5-style loop scoring consume Gate6 outputs without redefining the motif

Structure:

- same sample and step grouping spirit as `Gate4RunInputV1`
- provenance must clearly state:
  - source = Gate6-A native local span
  - vectors = local8 compatibility embedding
  - not original FWHT 8D proxy observables

Per step, required fields:

- `compat_vectors`
  - `V_local8`
  - `Splus_local8`
  - `Sminus_local8`
- labels and baseline fields copied through
- extra provenance field:
  - `boundary_origin = "gate6_native_local_span_local8_v1"`

Canonical Gate6-A compatibility artifacts must not overload legacy names such as `V_8d`, `Splus_8d`, and `Sminus_8d`.
If an existing Gate5 consumer cannot yet be parameterized away from those names, a separate derived adapter artifact may be emitted for that consumer only.
That adapter is not the canonical Gate6-A schema.

## 6. Scalar Summary Outputs

Scalar summaries are secondary artifacts only.

Recommended:

- token telemetry CSV
- sample summary CSV
- aggregate markdown report

But the primary artifact is the object family above.

## 7. Determinism Requirements

Gate6-A artifacts must be deterministic under fixed:

- model revision
- layer
- tokenizer
- input
- construction thresholds
- sign-fix rule

Re-run expectation:

- `manifest.json` may differ only in timestamps if timestamps are retained
- arrays and compatibility inputs must be bitwise-stable or documented to float tolerance if not feasible
- checksums must be emitted

## 8. Why This Schema Exists

The schema is designed so Gate6-A does not collapse back into a score-only system.

It keeps:

- the invariant object through the basis factor for `P_t`
- the local coordinates
- the raw local geometry summaries
- the benchmark-ready compatibility view

This is the minimal bridge between:

- native local observation design
- fair comparison against Gate5
- future downstream motif and field work
